import numpy as np
import scipy
import moviepy.editor as mpy


# This code allows gifs to be saved of the training episode for use in the Control Center.
def save_gif(frames, frame_shape, episode, rl='dqn', continuous=True):
    time_per_step = 0.05
    height, width = frame_shape[0], frame_shape[1]
    images = np.reshape(np.array(frames), [len(frames), height, width])
    if images.shape[1] != 3:
        if continuous:
            images = color_frame_continuous(images)
        else:
            images = color_frame(images)
    big_images = []
    for image in images:
        big_images.append(scipy.misc.imresize(image, [height * 10, width * 40], interp='nearest'))
    big_images = np.array(big_images)
    make_gif(big_images,
             '../../frames/%s/%d-%d/image' % (rl, height, width) + str(episode) + '.gif',
             duration=len(big_images) * time_per_step, true_image=True, salience=False)


def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS) / duration * t)]
        except:
            x = salIMGS[-1]
        return x

    txtClip = mpy.TextClip('.', color='white', font="Amiri-Bold",
                       kerning=5, fontsize=10)
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip = mpy.CompositeVideoClip([clip, txtClip])
    clip.duration = duration
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True, duration=duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps=len(images) / duration, verbose=False, logger=None)
        # clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps=len(images) / duration, verbose=False, logger=None)


def make_gif_with_count(images, counts, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS) / duration * t)]
        except:
            x = salIMGS[-1]
        return x
    clips = []
    num_frame = len(images)
    for f in range(num_frame):
        txtClip = mpy.TextClip(str(counts[f]), color='white', font="Amiri-Bold",
                           kerning=5, fontsize=10)
        _clip = mpy.ImageClip(images[f])
        _clip = mpy.CompositeVideoClip([_clip, txtClip])
        _clip.duration = duration/num_frame
        clips.append(_clip)
    clip = mpy.concatenate(clips)
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True, duration=duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps=len(images) / duration, verbose=False)
        # clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps=len(images) / duration, verbose=False)


def color_frame(images, dim=2):
    color_map = {
        0: [0, 0, 0],  # black
        1: [0, 255, 0],  # green
        2: [0, 0, 255],  # blue
        3: [255, 0, 0],  # red
    }
    if dim == 2:
        colored_images = np.zeros([len(images), images.shape[1], images.shape[2], 3])
        for k in range(len(images)):
            for i in range(images.shape[1]):
                for j in range(images.shape[2]):
                    colored_images[k, i, j] = color_map[int(images[k, i, j])]
    return colored_images


def color_frame_continuous(images, dim=2):
    if dim == 2:
        colored_images = np.zeros([len(images), images.shape[1], images.shape[2], 3])
        for k in range(len(images)):
            for i in range(images.shape[1]):
                for j in range(images.shape[2]):
                    if images[k, i, j] == -1.0:
                        colored_images[k, i, j] = [0, 0, 0]
                    else:
                        grey = max(0, 255 - 0.1 * 255 * images[k, i, j])
                        colored_images[k, i, j] = [grey, grey, grey]
    return colored_images