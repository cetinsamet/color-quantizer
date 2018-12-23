import sys
from ColorQuantizer import ColorQuantizer


def main(argv):

    if len(argv) != 1:
        print("Usage: python quantize_image.py imagepath")
        exit()

    image_path	= argv[0]
    weight_path	= image_path.split('.')[0] + ".txt"
    save_path	= image_path.split('.')[0] + ".png"

    quantizer 	= ColorQuantizer(n_colors=2, random_state=0)
    quantizer.quantize_image(image_path, weight_path, save_path)
    print("-> quantization is completed.")
    return

if __name__ == '__main__':
    main(sys.argv[1:])	
	