import cv2

def plot_img_and_mask(plt, ax, img, mask, img_new_bg, i):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    ax[0].set_title('Input image ' + str(i))
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask ' + str(i))
        ax[1].imshow(mask)
        ax[2].set_title(f'New Input image ' + str(i))
        ax[2].imshow(cv2.cvtColor(img_new_bg, cv2.COLOR_BGR2RGB))

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.pause(0.001)
