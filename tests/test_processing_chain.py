# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import sys
import os
import cv2
import numpy as np
import multiprocessing
import time
import queue

# Ensure lib is importable
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from aglaia.ImageBuffer import ImageBuffer, ImageType
from aglaia.workers.IntegratedProcessingChain import IntegratedProcessingChain
from aglaia.workers.chain_abstraction import SimpleChainElement
from aglaia.processors.SkewFinder import SkewFinderOption

def test_chain():
    print("Testing IntegratedProcessingChain...")
    
    # Setup Log Queue
    log_queue = multiprocessing.Queue()
    
    # 1. Chain Config
    skew_opts = SkewFinderOption(max_angle=5.0, min_angle=0.1, apply_rotation=True)
    elements = [
        SimpleChainElement("SkewFinder", skew_opts)
    ]
    
    # 2. Init Chain
    try:
        # elements list, num_workers, log_queue, queue_factory
        chain = IntegratedProcessingChain(elements, 1, log_queue)
        chain.start()
        print("Chain started.")
    except Exception as e:
        print(f"Failed to start chain: {e}")
        return

    # 3. Create Dummy Input
    # 300x300 white image with a black line rotated
    img = np.ones((300, 300), dtype=np.uint8) * 255
    # Draw a line that is slightly skewed (e.g. 45 degrees)
    cv2.line(img, (50, 50), (250, 250), 0, 5)
    
    # Add dummy out_dir
    os.makedirs("test_output/raw", exist_ok=True)
    os.makedirs("test_output/deskewed", exist_ok=True)
    
    ib = ImageBuffer(img, ImageType.BW, dpi=72, path="test.png", filestem="test_skew", out_dir="test_output/raw")
    
    # 4. Enqueue
    chain.enqueue(ib)
    
    # 5. Monitor Output
    # SkewFinder output is persisted by the chain.
    
    start_t = time.time()
    success = False
    while time.time() - start_t < 5:
        try:
            msg = log_queue.get(timeout=0.5)
            # print(f"Log: {msg}")
            if msg[0] == 'image_event':
                 # ('image_event', raw_path, filestem, type, full_path, meta)
                 etype = msg[3]
                 print(f"Received Image Event: {etype}")
                 if etype == 'deskewed':
                     print("Success: Deskew event received.")
                     success = True
                     break
        except queue.Empty:
            continue
            
    chain.stop()
    
    if success:
        print("Test Passed.")
    else:
        print("Test Failed: No output event.")

if __name__ == "__main__":
    test_chain()
