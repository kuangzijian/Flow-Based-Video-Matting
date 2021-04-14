import cv2

def plot_img_and_mask(plt, ax, img, mask, removed_bg, img_new_bg):
    ax[0, 0].clear()
    ax[0, 1].clear()
    ax[1, 0].clear()
    ax[1, 1].clear()

    ax[0, 0].set_title('Input frame')
    ax[0, 0].imshow(img)
    ax[0, 1].set_title(f'Predicted mask')
    ax[0, 1].imshow(mask)
    ax[1, 0].set_title(f'Background removed')
    ax[1, 0].imshow(cv2.cvtColor(removed_bg, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title(f'Background replaced')
    ax[1, 1].imshow(cv2.cvtColor(img_new_bg, cv2.COLOR_BGR2RGB))

    for a in ax:
        for aa in a:
            aa.set_xticks([])
            aa.set_yticks([])
    plt.pause(0.0001)