import imageio.v2 as imageio  # используем v2 для совместимости
import os

frames_dir = 'output_frames'  # директория с фреймами
output_video_path = 'hole_sonic.mp4'  # путь к выходному видео
fps = 144  # желаемая частота кадров

# получаем список PNG-файлов и сортируем по номеру
frame_filenames = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
frame_filenames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# читаем изображения и собираем в видео
with imageio.get_writer(output_video_path, fps=fps, codec='libx264', quality=10) as writer:
    for frame_filename in frame_filenames:
        frame_path = os.path.join(frames_dir, frame_filename)
        image = imageio.imread(frame_path)
        writer.append_data(image)

print(f'Video saved as {output_video_path} with {fps} FPS')
