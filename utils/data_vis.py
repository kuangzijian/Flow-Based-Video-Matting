import matplotlib.pyplot as plt
import cv2

def plot_img_and_mask(img, mask, img_new_bg):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, 3, figsize=(12,3))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
        ax[2].set_title(f'Input image after replacing background')
        ax[2].imshow(cv2.cvtColor(img_new_bg, cv2.COLOR_BGR2RGB))

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()
