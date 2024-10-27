import os

def verify_cityscapes_structure():
    root_dir = "D:/Research Internship IISER Bhopal/proto-segmentation-master sacha/proto-segmentation-master/CityScapes Dataset"
    
    # Check leftImg8bit structure
    left_img_dir = os.path.join(root_dir, 'leftImg8bit_trainvaltest', 'train')
    if not os.path.exists(left_img_dir):
        print("❌ leftImg8bit/train directory not found!")
        return False
    
    # Check for at least one city directory
    cities = os.listdir(left_img_dir)
    if not cities:
        print("❌ No city directories found in leftImg8bit/train!")
        return False
    
    print(f"✓ Found {len(cities)} cities in leftImg8bit/train:")
    for city in cities:
        city_path = os.path.join(left_img_dir, city)
        if os.path.isdir(city_path):
            image_files = [f for f in os.listdir(city_path) if f.endswith('_leftImg8bit.png')]
            print(f"  - {city}: {len(image_files)} images")
    
    # Count total images
    total_images = 0
    for city in cities:
        city_path = os.path.join(left_img_dir, city)
        if os.path.isdir(city_path):
            image_files = [f for f in os.listdir(city_path) if f.endswith('_leftImg8bit.png')]
            total_images += len(image_files)
    
    print(f"\nTotal images found: {total_images}")
    return total_images > 0

if __name__ == "__main__":
    print("Verifying Cityscapes dataset structure...")
    if verify_cityscapes_structure():
        print("\n✓ Dataset structure appears correct!")
    else:
        print("\n❌ Dataset structure is incomplete or incorrect!")
        print("\nPlease ensure you have:")
        print("1. Downloaded leftImg8bit_trainvaltest.zip from the Cityscapes website")
        print("2. Extracted the ZIP file completely")
        print("3. Copied the contents to the correct location")