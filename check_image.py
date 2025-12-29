import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def verify_and_clean_images(src_dir, dst_dir):
    """
    遍历 src_dir，校验图片完整性，统一转为 RGB 并保存到 dst_dir。
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # 如果目标文件夹不存在，则创建
    if not dst_path.exists():
        dst_path.mkdir(parents=True)
        print(f"创建目标文件夹: {dst_path}")

    # 获取所有图片文件 (支持 jpg, png 等)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in src_path.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"开始校验并重存图片，共计: {len(image_files)} 张")
    
    bad_count = 0
    success_count = 0

    for img_file in tqdm(image_files, desc="处理进度"):
        try:
            # 1. 尝试打开图片
            with Image.open(img_file) as img:
                # 2. 验证图片数据是否损坏
                img.verify() 
            
            # verify() 后需要重新打开才能操作
            with Image.open(img_file) as img:
                # 3. 统一转换为 RGB 格式 (去除 Alpha 通道或转灰度为彩图)
                # 这对 SAM 模型的输入至关重要
                img_rgb = img.convert('RGB')
                
                # 4. 保存到新文件夹 (保持原文件名)
                target_file = dst_path / img_file.name
                img_rgb.save(target_file, "JPEG", quality=95)
                success_count += 1
                
        except Exception as e:
            print(f"\n[错误] 图片损坏或无法处理: {img_file.name} | 原因: {e}")
            bad_count += 1

    print(f"\n处理完成！")
    print(f"成功保存: {success_count} 张")
    print(f"识别并跳过坏图: {bad_count} 张")
    print(f"清洗后的数据已存储在: {dst_dir}")


if __name__ == "__main__":
    # --- 请根据你的实际路径修改以下变量 ---
    SOURCE_FOLDER = "/data/LATEX-EnxText-new/train/images"
    TARGET_FOLDER = "path/to/your/cleaned/images"
    
    verify_and_clean_images(SOURCE_FOLDER, TARGET_FOLDER)