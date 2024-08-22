import cv2
import numpy as np
import os

def cylindrical_warp(img, K=0.5):
  """
  Performs cylindrical warping on a image

  Args:
      img: Input image (numpy array)
      K: Cyindrical factor  

  Returns:
      Warped Image
  """

  h, w = img.shape[:2]
  center = (w/2, h/2)

  # Generate grid coordinates
  x = np.linspace(0, w-1, w)
  y = np.linspace(0, h-1, h)
  X, Y = np.meshgrid(x,y)

  # Calculate cylindrical coordinates
  r = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
  theta = np.arctan2(Y - center[1], X - center[0])

  # Project to cylindrical coordinates
  X_cyl = r * np.cos(theta * K) + center[0]
  Y_cyl = r * np.sin(theta * K) + center[1]

  # Interpolate the image
  warped_img = cv2.remap(img, X_cyl.astype(np.float32), Y_cyl.astype(np.float32), cv2.INTER_LINEAR)

  return warped_img

# Example usage
img = cv2.imread("Footages/Image/sample.png")
warped = cylindrical_warp(img, K=0.5)
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()