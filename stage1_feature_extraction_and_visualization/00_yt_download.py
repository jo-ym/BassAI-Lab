import csv
from yt_dlp import YoutubeDL


def download_full_channel(channel_url, output_dir="downloads", csv_file="metadata.csv"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'getinfo': True, 
    }
    
    metadata_list = []
    
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(channel_url, download=True) 
        
        if 'entries' in info_dict:
            for entry in info_dict['entries']:
                if not entry:
                    continue
                metadata_list.append({
                    "title": entry.get('title'),
                    "id": entry.get('id'),
                    "url": entry.get('webpage_url'),
                    "duration": entry.get('duration'),
                    "upload_date": entry.get('upload_date'),
                    "view_count": entry.get('view_count'),
                    "like_count": entry.get('like_count'),
                    "channel": entry.get('channel'),
                    "tags": ', '.join(entry.get('tags', [])),
                })
    
    with open(csv_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metadata_list[0].keys())
        writer.writeheader()
        writer.writerows(metadata_list)
    
    print(f"Metadata saved to {csv_file}")


download_full_channel(
    channel_url="",
    output_dir="stage1/downloads",
    csv_file="stage1/metadata.csv"
)
