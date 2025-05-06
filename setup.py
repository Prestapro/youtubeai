from setuptools import setup

setup(
    name='youtubeai-extract',
    version='0.1.0',
    description='Extract clips from videos using Whisper and MoviePy.',
    author='Prestapro & ChatGPT',
    author_email='your@email.com',
    py_modules=['extract_clips_by_keyword'],
    install_requires=[
        'openai-whisper',
        'moviepy',
        'ffmpeg-python',
        'srt',
        'numpy',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'extract-clips=extract_clips_by_keyword:main',
        ],
    },
    python_requires='>=3.8',
)