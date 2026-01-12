import os
import logging
from typing import Type, Dict, Any, Optional

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(
        self, 
        model_name: str,
        api_key: Optional[str] = None,  # 允许为 None
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
        temperature: float = 0
    ):
        """
        初始化 LLM 客户端
        
        :param api_key: API Key。如果为 None，尝试从环境变量 DASHSCOPE_API_KEY 读取。
        :param model_name: 模型名称。
        """
        # 1. 优先使用参数，其次使用环境变量
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API Key 未找到。请通过参数传入，或设置环境变量 'DASHSCOPE_API_KEY'。"
            )

        # 2. 初始化 ChatOpenAI
        # 超时时间可以通过环境变量配置，默认90秒（对于大多数请求足够）
        # 如果处理大文件或复杂任务，可以通过环境变量增加：export LLM_TIMEOUT=120
        request_timeout = int(os.getenv("LLM_TIMEOUT", "90"))  # 默认90秒
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=self.api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_retries=1,  # 最多重试1次
            request_timeout=request_timeout,  # 可配置的超时时间
        )
        
        logger.info(f"LLM Client timeout set to {request_timeout} seconds")
        
        # 注意：重试信息 "Retrying request" 来自 langchain_openai 的底层 HTTP 客户端
        # 重试通常发生在：网络超时、API 5xx错误、连接错误、API限流(429)等情况
        
        logger.info(f"LLM Client initialized. Model: {model_name}")

    # ... create_json_chain, create_text_chain, invoke 方法保持不变 ...
    def create_json_chain(self, pydantic_model: Type[BaseModel], system_prompt: str, user_prompt_template: str) -> Runnable:
        parser = JsonOutputParser(pydantic_object=pydantic_model)
        final_user_prompt = user_prompt_template
        if "{format_instructions}" not in final_user_prompt:
            final_user_prompt += "\n\n请严格遵循以下 JSON 输出格式：\n{format_instructions}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", final_user_prompt),
        ]).partial(format_instructions=parser.get_format_instructions())
        return prompt | self.llm | parser

    def create_text_chain(self, system_prompt: str, user_prompt_template: str) -> Runnable:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt_template),
        ])
        return prompt | self.llm | StrOutputParser()

    def invoke(self, chain: Runnable, inputs: Dict[str, Any]) -> Optional[Dict]:
        """
        同步调用 LLM
        """
        try:
            return chain.invoke(inputs)
        except Exception as e:
            logger.error(f"LLM Invocation Error: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    async def ainvoke(self, chain: Runnable, inputs: Dict[str, Any]) -> Optional[Dict]:
        """
        异步调用 LLM，带重试机制
        """
        import asyncio
        max_attempts = int(os.getenv("LLM_MAX_RETRIES", "1"))  # 应用层不重试，仅底层重试（底层会重试1次）
        base_wait_time = float(os.getenv("LLM_RETRY_WAIT", "2"))  # 基础等待时间（秒）
        
        for attempt in range(max_attempts):
            try:
                result = await chain.ainvoke(inputs)
                if result is not None:
                    return result
                # 如果返回None，可能是API返回了空结果，也记录警告
                if attempt < max_attempts - 1:
                    logger.warning(f"LLM调用返回None（尝试 {attempt + 1}/{max_attempts}），将重试")
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # 如果是最后一次尝试，记录详细错误
                if attempt == max_attempts - 1:
                    logger.error(
                        f"LLM Async Invocation Error (最终失败，已重试{max_attempts}次): "
                        f"{error_type}: {error_msg}"
                    )
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    return None
                
                # 计算等待时间（指数退避）
                wait_time = base_wait_time * (2 ** attempt)
                logger.warning(
                    f"LLM调用失败（尝试 {attempt + 1}/{max_attempts}）: {error_type}: {error_msg}，"
                    f"{wait_time:.1f}秒后重试..."
                )
                await asyncio.sleep(wait_time)
        
        return None