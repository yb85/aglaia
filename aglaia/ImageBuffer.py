# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

import os
import cv2
import numpy as np
import math
from pathlib import Path
from PIL import Image
from enum import Enum

class ImageType(Enum):
    BW = "BW"
    GRAY = "GRAY"
    COLOR = "COLOR"

class ImageBuffer:
    """Standardized object for image data exchange."""
    def __init__(self, buffer, type: ImageType, dpi: float = 72.0, path: str = None, parent: 'ImageBuffer' = None, filestem: str = None, out_dir: str = None, parent_stem: str = None,
                 scan_id: int = None, parent_node_id: int = None, pipeline_version_id: int = None,
                 branch_path: str = "", branch_label: str = None, depth: int = 0):
        self.buffer = buffer
        self.type = type
        self.dpi = dpi
        self.path = path
        self.parent = parent
        self.filestem = filestem
        self.out_dir = out_dir

        # Tree context (storage layer, M0)
        self.scan_id = scan_id
        self.parent_node_id = parent_node_id
        self.pipeline_version_id = pipeline_version_id
        self.branch_path = branch_path
        self.branch_label = branch_label
        self.depth = depth

        # Explicitly store parent stem string to avoid pickling heavy parent objects unnecessarily
        # or losing track if parent is None after copy
        self.parent_stem = parent_stem
        if not self.parent_stem and self.parent:
             if isinstance(self.parent, ImageBuffer):
                 self.parent_stem = self.parent.filestem
             else:
                 # Attempt to deduce if parent is string path?
                 try:
                      self.parent_stem = Path(str(self.parent)).stem
                 except: pass

        self.children = [] # List of child ImageBuffers (e.g. layouts)
        self.meta = {} # Metadata dict (ROI, OCR text, etc.)

    def __repr__(self):
        return f"<ImageBuffer {self.type.value} {self.buffer.shape} @ {self.dpi}dpi>"

    def __getstate__(self):
        # Strip live parent/children refs before pickling: a 2 MB layout
        # crop would otherwise drag its full-res parent frame (and that
        # frame's children, recursively) across every queue hop. Lineage
        # survives via parent_stem + parent_node_id/scan_id.
        state = self.__dict__.copy()
        state["parent"] = None
        state["children"] = []
        return state

    def to_rgb(self):
        """Convert buffer to 3-channel RGB numpy array."""
        img = self.buffer
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

    def copy(self):
        """Create a deep clone of this buffer for safe async processing/writing."""
        import copy
        new_buf = ImageBuffer(
            buffer=self.buffer.copy() if self.buffer is not None else None,
            type=self.type,
            dpi=self.dpi,
            path=self.path,
            parent=self.parent, # Note: parent usually stays as ref or string
            filestem=self.filestem,
            out_dir=self.out_dir,
            parent_stem=self.parent_stem, # COPY THE STRING LINK
            scan_id=self.scan_id,
            parent_node_id=self.parent_node_id,
            pipeline_version_id=self.pipeline_version_id,
            branch_path=self.branch_path,
            branch_label=self.branch_label,
            depth=self.depth,
        )
        new_buf.meta = copy.deepcopy(self.meta)
        return new_buf

    def to_gray(self):
        """Convert buffer to 1-channel grayscale numpy array (0-255)."""
        img = self.buffer
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def to_bw(self):
        """Convert buffer to 1-channel binary numpy array (0 or 255)."""
        gray = self.to_gray()
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return bw

    def check_binary(self):
        """Check if buffer is likely binary (few unique values)."""
        img = self.buffer
        if len(img.shape) == 2:
            from aglaia.processors.utils import count_distinct_values
            return count_distinct_values(img) <= 2
        return False

    def rescale(self, target_dpi, threshold=0.01):
        """Rescale image to target DPI."""
        if target_dpi and self.dpi != target_dpi and self.dpi > 0:
            scale_factor = target_dpi / self.dpi
            if abs(scale_factor - 1.0) > threshold:
                img = self.buffer
                nh = int(img.shape[0] * scale_factor)
                nw = int(img.shape[1] * scale_factor)
                if nh > 0 and nw > 0:
                     self.buffer = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
                     self.dpi = target_dpi
        return self


    def binarize(self, processor_or_queue):
        """
        Apply binarization or queue for binarization.
        processor_or_queue: either a Binarizer instance or a Queue instance.
        """
        # Check if it's a queue (duck typing)
        if hasattr(processor_or_queue, 'put'):
             processor_or_queue.put(self)
             return self
        
        # Assume it's a Binarizer object with a binarize method
        if hasattr(processor_or_queue, 'binarize'):
             processor_or_queue.binarize(self)
             return self
             
        return self

    def detectLayouts(self, processor_or_queue):
        """
        Apply layout detection or queue for detection.
        processor_or_queue: either a PageDetector instance or a Queue instance.
        """
        if hasattr(processor_or_queue, 'put'):
             processor_or_queue.put(self)
             return self
        
        if hasattr(processor_or_queue, 'process'):
             # Note: This assumes processor_or_queue.process() can handle self as first arg
             # or is suitably bound. For PageDetector.process, it takes options etc.
             # This method is primarily valid if the processor is fully configured 
             # or if we are just queuing. 
             # Ideally PageDetector.process should have a simpler signature for this valid standard usage.
             # But for duck typing compliance requested:
             processor_or_queue.process(self)
             return self
             
        return self

        return self

    def dewarp(self, processor_or_queue):
        """
        Apply dewarping or queue for dewarping.
        processor_or_queue: either a PageDewarper instance or a Queue instance.
        """
        if hasattr(processor_or_queue, 'put'):
             processor_or_queue.put(self)
             return self
        
        if hasattr(processor_or_queue, 'dewarp'):
             processor_or_queue.dewarp(self)
             return self
             
        return self

    def write(self, collection, options, suffix=None, log_queue=None, event_type=None, executor=None):
        """
        Handles standardized image persistence and optional event signaling.
        collection: 'raw', 'layout', 'dewarp', 'debug' or any custom instance name
        options: dict containing 'paths' and 'general'
        suffix: optional suffix for filename
        executor: optional ThreadPoolExecutor for async writing
        """
        paths = options.get("paths", {})
        general = options.get("general", {})
        
        # Decide base directory
        base_path = None
        
        # 1. Check if explicitly mapped in paths
        if collection in paths and paths[collection]:
            base_path = Path(paths[collection])
            
        # 2. Check for Workspace Root (The folder passed as argument)
        elif 'root' in paths:
            # FORCE intermediate folders to project root
            base_path = Path(paths['root']) / collection
            
        # 3. Fallback to existing out_dir logic (used in PDF processing etc)
        elif self.out_dir:
            base_path = Path(self.out_dir)
            # If we are in or under a standard folder structure, use sibling for collections
            output_root = Path(paths.get('output', '.')).resolve()
            current_base = base_path.resolve()
            
            if current_base == output_root:
                 # If we are purely at the output root and no project root was provided,
                 # we have no choice but to nest or go to parent.
                 # User wants NO nesting in XX_OUTPUT, so let's go up if possible.
                 base_path = base_path.parent / collection
            if collection == "output":
                 # Final output respects out_dir exactly as set
                 pass 
            elif base_path.name in ['layout', 'raw', 'dewarp', 'debug'] or base_path.name != collection:
                 # Intermediate files go to siblings of the output folder
                 base_path = base_path.parent / collection
        
        if base_path is None:
            # Final fallback: workspace root if possible, else current output parent
            root = paths.get('root')
            if root:
                base_path = Path(root) / collection
            else:
                base_path = Path(paths.get('output', '.')).parent / collection
        
        if base_path is None:
            base_path = Path(paths.get('output', '.')) / collection
            
        # Filestem: prefer buffer's filestem, then paths['filestem'], then default
        filename = self.filestem or paths.get('filestem', 'capture')
        if suffix:
            filename += f"_{suffix}"
            
        ext = ".png" if self.type == ImageType.BW else ".jpg"
        full_path = base_path / (filename + ext)

        overwrite = options.get('overwrite', paths.get('overwrite', general.get('overwrite', True)))
        
        if not overwrite and full_path.exists():
            # Still signal if requested even if file exists
            if log_queue and event_type:
                 self._emit_event(log_queue, event_type, str(full_path))
            return full_path

        # UNIVERSAL CLEANUP: Remove any existing file with the same stem but different extension
        # to prevent duplicate JPG/PNG accumulation.
        if base_path.exists():
            for alt_ext in ['.jpg', '.jpeg', '.png']:
                if alt_ext == ext: continue
                alt_path = base_path / (filename + alt_ext)
                if alt_path.exists():
                    try:
                        alt_path.unlink()
                    except: pass

        # Prepare I/O function
        def _io_task(buffer, path, dpi, type_):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                
                dpi_val = (dpi, dpi) if not isinstance(dpi, tuple) else dpi
                
                if str(path).endswith(".jpg"):
                    pil_img = Image.fromarray(buffer)
                    if type_ == ImageType.GRAY:
                        pil_img = pil_img.convert('L')
                    else:
                        pil_img = pil_img.convert('RGB')
                    pil_img.save(path, dpi=dpi_val, quality=95, optimize=True)
                else:
                    # PNG
                    if type_ == ImageType.BW:
                        pil_img = Image.fromarray(buffer).convert('1')
                        pil_img.save(path, dpi=dpi_val, optimize=True)
                    else:
                        out_img = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR) if len(buffer.shape) == 3 else buffer
                        cv2.imwrite(str(path), out_img)
                
                # Signal after write
                if log_queue and event_type:
                     self._emit_event(log_queue, event_type, str(path))
            except Exception as e:
                import traceback
                traceback.print_exc()
                if log_queue:
                     log_queue.put(('error', f"Async Write Error ({path.name}): {e}"))

        # Execute
        if executor:
            # Copy buffer for thread safety as pipeline might mutate it next
            buf_copy = self.buffer.copy()
            executor.submit(_io_task, buf_copy, full_path, self.dpi, self.type)
        else:
            _io_task(self.buffer, full_path, self.dpi, self.type)

        return full_path

    def _emit_event(self, log_queue, event_type, full_path):
        """Internal helper to format and send image_event to GUI."""
        import os
        from pathlib import Path

        # Map raw_path (root parent)
        curr = self
        raw_path = self.path
        parent_stem = None
        
        if self.parent_stem:
            parent_stem = self.parent_stem
            # Still try to find root raw_path if possible
            if self.parent and isinstance(self.parent, ImageBuffer):
                 curr = self.parent
                 while curr.parent and isinstance(curr.parent, ImageBuffer):
                      curr = curr.parent
                 raw_path = curr.path
        elif self.parent:
            if isinstance(self.parent, ImageBuffer):
                parent_stem = self.parent.filestem
                # Recurse to find root if needed
                p_curr = self.parent
                while p_curr.parent and isinstance(p_curr.parent, ImageBuffer):
                    p_curr = p_curr.parent
                raw_path = p_curr.path
            else:
                # Parent is likely a string path
                parent_stem = Path(str(self.parent)).stem
                raw_path = str(self.parent)

        if raw_path is None:
            raw_path = self.path or "buffer"

        gui_meta = self.meta.copy() if self.meta else {}

        # print(f"DEBUG: Putting image_event for {self.filestem} (raw={raw_path}, parent={parent_stem})")
        log_queue.put(('image_event', 
                       os.path.normpath(os.path.abspath(str(raw_path))), 
                       self.filestem, 
                       event_type, 
                       os.path.normpath(os.path.abspath(str(full_path))), 
                       gui_meta,
                       parent_stem))


