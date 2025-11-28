# EE6434 Data Driven Control - Course Project

DMD based Real-Time Background/Foreground Separation in Video

## Overview

This project implements Dynamic Mode Decomposition (DMD) for real-time background/foreground separation in video streams.

Based on the paper:

- [**Dynamic Mode Decomposition for Real-Time Background/Foreground Separation in Video, J. Grosek and J. Nathan Kutz**](https://arxiv.org/abs/1404.7592)

A few minor adjustments made by me.

## Project Description

Dynamic Mode Decomposition (DMD) is a data-driven approach for analyzing complex dynamical systems. This implementation focuses on applying DMD techniques to separate background and foreground elements in video streams in real-time.

## Setup Instructions

### Prerequisites

- Python 3.8
- pip (Python package manager)

### Creating a Virtual Environment

**Using venv:**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # macOS/Linux
```

**Using conda:**

```bash
conda create -n dmd-env python=3.8
conda activate dmd-env
```

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

```bash
python camera.py
```

## Demo

A demonstration video showing the real-time background/foreground separation:

- [Watch Demo Video](./demo.mkv)
