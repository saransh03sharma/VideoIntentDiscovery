import youtube_dl
import multiprocessing
import progressbar
import os
import json
import traceback
import sys
from collections import defaultdict
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm
from moviepy.editor import VideoFileClip

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root directory path to sys.path
sys.path.append(root_path)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = '0'

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download_video_with_youtube_dl(url, output_path):
    ydl_opts = {
        'format': 'best',  # Best available quality
        'outtmpl': output_path,  # Output file path
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def get_slices(url, timestamps, out_dir):
    timestamps.sort(key=lambda y: y[0])
    tmp_name = os.path.join(out_dir, url.replace("/", "__") + ".mp4")
    print("Downloading %s" % url)
    try:
        if not os.path.exists(tmp_name):
            download_video_with_youtube_dl(url, tmp_name)
        
        print("Slicing for %s timestamps" % len(timestamps))
        video = VideoFileClip(tmp_name)
        
        for timestamp, windows in timestamps:
            for window in windows:
                start_time = max(0, timestamp - window)
                end_time = timestamp + window
                span = window * 2
                
                output_filename = tmp_name[:-4] + ".%s.cut.%s.mp4" % (int(timestamp), span)
                ffmpeg_extract_subclip(tmp_name, start_time, end_time, targetname=output_filename)
        os.remove(tmp_name)
    except Exception as e:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        with open('error_log.txt', 'a+') as f:
            f.write(url + '\n')
            traceback.print_exc(file=f)
            f.write('----------------------------------------------------------' + '\n')

def get_videos(video_ids, video2timestamps, out_dir):

    for url in tqdm(video_ids):
        get_slices(url, video2timestamps[url], out_dir)

def download_videos_for_intent_dataset(data_dir, out_dir, n_processes=3):

    files = [os.path.join(data_dir, 'train.json'),
             os.path.join(data_dir, 'dev.json'),
             os.path.join(data_dir, 'test.json')]
    samples_by_video = defaultdict(lambda: [])
    exists = 0
    for f in files:
        dataset = json.load(open(f, 'r'))
        for sample in dataset:
            video_id = sample["video_id"]
            # Construct the video URL with ".mp4" extension
            video_url = video_id
            timestamp = float(sample["timestamp"])
            tmp_name = os.path.join(out_dir, video_id.replace("/", "__") + ".mp4")
            if not tmp_name.endswith('.mp4'):
                tmp_name += '.mp4'
            assert tmp_name.endswith('.mp4'), tmp_name
            target_names = [tmp_name[:-4] + ".%s.cut.%s.mp4" % (int(timestamp), t) for t in [10, 20]]
            if all([os.path.exists(fname) for fname in target_names]):
                exists += 1
                continue
            else:
                samples_by_video[video_url].append((float(timestamp), [5, 10]))
    
    print("Queued %s videos for download" % len(samples_by_video))
    print("Start %s jobs" % n_processes)
    video_ids = list(samples_by_video.keys())
    total_videos = sum([len(samples_by_video[k]) for k in video_ids])
    print("%s clips exist" % exists)
    print("Queued %s clips for splicing" % total_videos)

    jobs = []
    for i in range(0, n_processes):
        process = multiprocessing.Process(target=get_videos,
                                          args=(video_ids[int(i*len(video_ids)/n_processes):int((i+1)*len(video_ids)/n_processes)], samples_by_video, out_dir))
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()

# Call the function to download and process videos
download_videos_for_intent_dataset('../data/bid/', '../data/bid/video', n_processes=8)
