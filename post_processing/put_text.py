import numpy as np
import cv2
import imageio
import pathlib

def get_episode_nbr(file_path):
  name = file_path.stem
  nbr = name[-6:] # Lazy way to get the number at the end
  return int(nbr)


def get_images(file_path):
  cap = cv2.VideoCapture(str(file_path))
  e_nbr = get_episode_nbr(file_path)

  images = []
  while(cap.isOpened()):
      ret, frame = cap.read()

      if ret:
        image = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        img_height = image.shape[1]
        margin = 20

        cv2.putText(img = image,
                    text = 'Episode: {}'.format(e_nbr),
                    org = (margin, img_height - margin),
                    fontFace = cv2.FONT_HERSHEY_COMPLEX,
                    fontScale = 1,
                    thickness = 1,
                    color = (0, 0, 0))

        images.append(image)
      else:
        break

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()

  return images


if __name__ == "__main__":
  images = []

  dir_path = pathlib.Path('../saves/last_run/')

  for file_path in dir_path.glob("*.json"):
    file_path.unlink()

  for file_path in dir_path.glob("*.mp4"):
    images.extend(get_images(file_path))
    # file_path.unlink() # TODO: Add in again?

  imageio.mimsave('{}/replay.gif'.format(dir_path), images, duration=0.03, subrectangles=True)
