import cv2
import os
import random

def process_images():
    # ================= 配置区域 =================
    # 1. 输入图片的文件夹路径
    input_dir = "/home/chenyue001/papers/images/Test_image_image1"
    
    # 2. 目标基础路径 (在这里面新建文件夹)
    revert_base_dir = "/home/chenyue001/papers/images_revert"
    
    # 3. 新建的文件夹名字
    # 最终保存路径将是: /home/chenyue001/papers/images_revert/Test_image_image1_rotated
    new_folder_name = "Test_image_image1_rotated"
    # ===========================================

    # 拼接最终输出路径
    output_dir = os.path.join(revert_base_dir, new_folder_name)

    # 1. 创建输出目录 (如果路径不存在，makedirs 会自动创建多级目录)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 已新建输出文件夹: {output_dir}")
    else:
        print(f"📂 输出文件夹已存在: {output_dir}")

    # 2. 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 错误: 找不到输入文件夹 {input_dir}")
        return

    # 3. 定义旋转角度映射 (OpenCV 常量)
    rotation_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    
    # 支持的图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    print(f"🚀 开始处理图片...")
    print(f"   源路径: {input_dir}")
    print(f"   目标路径: {output_dir}")
    print("-" * 50)

    count = 0
    files = os.listdir(input_dir)

    for filename in files:
        file_path = os.path.join(input_dir, filename)

        # 跳过文件夹
        if not os.path.isfile(file_path):
            continue

        # 检查是否为图片格式
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        # 读取图片
        img = cv2.imread(file_path)
        if img is None:
            print(f"⚠️  警告: 无法读取 {filename}")
            continue

        # --- 核心算法：随机旋转 ---
        angle = random.choice([90, 180, 270])
        try:
            rotated_img = cv2.rotate(img, rotation_map[angle])
            
            # 保存图片到新文件夹
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, rotated_img)
            
            count += 1
            if count % 50 == 0:
                print(f"   已处理 {count} 张...")

        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")

    print("-" * 50)
    print(f"✅ 处理完成！共生成 {count} 张旋转后的图片。")
    print(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    process_images()