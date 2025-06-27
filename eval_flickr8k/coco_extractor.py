#!/usr/bin/env python3
"""
Utility functions for working with the COCO dataset.
"""
from pycocotools.coco import COCO
from pathlib import Path
from typing import List, Dict, Generator, Union
from tqdm import tqdm
import json

class COCOCaptionExtractor:
    def __init__(self, 
                 annotations_path:Path, 
                 images_path:Path
                 )-> None:
        """
        Init COCOCaptionExtractor
        Args:
            annotations_path: Path to COCO annotations directory
            images_path: Path to COCO images directory
        """
        self.annotations_path = Path(annotations_path)
        self.images_path = Path(images_path)
        
        # Load COCO annotations
        captions_file = self.annotations_path / "captions_train2014.json"
        if not captions_file.exists():
            raise FileNotFoundError(f"Captions file not found: {captions_file}")
            
        print(f"Loading COCO annotations from {captions_file}...")
        self.coco = COCO(str(captions_file))
        print(f"Loaded {len(self.coco.imgs)} images with annotations")
    
    def len_files(self):
        """
        Returns:
            Number of images
        """
        return len(self.coco.imgs)
    
    def len_captions(self):
        """
        Returns:
            Number of captions
        """
        total_captions = 0
        for img_id in self.coco.imgs.keys():
            total_captions += len(self.coco.getAnnIds(imgIds=img_id))
        return total_captions
    
    def extract_captions(self, 
                         max_images:int=None,
                         )-> List:
        """
        Extract image-caption pairs
        
        Args:
            max_images: Maximum number of images to process (None for all)
        
        Returns:
            List of dicts with image info and captions
        """
        print("Extracting captions...")
        
        image_ids = list(self.coco.imgs.keys())
        if max_images:
            image_ids = image_ids[:max_images]
        
        extracted_data = []
        
        for img_id in tqdm(image_ids, desc="Processing images"):
            # Get image info
            img_info = self.coco.imgs[img_id]
            image_path = self.images_path / img_info['file_name']
            
            if not image_path.exists():
                continue
            
            # Get all captions for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Extract captions
            captions = [ann['caption'] for ann in anns]
            
            # Create data entry
            data_entry = {
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'image_path': str(image_path),
                'width': img_info['width'],
                'height': img_info['height'],
                'captions': captions,
                'num_captions': len(captions)
            }
            
            extracted_data.append(data_entry)
        
        print(f"Extracted captions for {len(extracted_data)} images")
        return extracted_data
    
    def save_data(self, 
                  data, 
                  output_path, 
                  format='json'):
        """Save extracted captions to json file"""
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved to {output_path}")
        else:
            raise ValueError("Supported formats: 'json'")
    
    def iter_image_captions(self, 
                            max_images: int = None, 
                            output_format: str = 'dict'
                            ) -> Generator[Union[Dict, str], None, None]:
        """
        Iterator that yields image path and captions
        
        Args:
            max_images: Maximum number of images to process (None for all)
            output_format: 'dict' returns dict, 'json' returns JSON string
        
        Yields:
            dict or JSON string with image_path and captions
        """
        print("Iterating through image-caption pairs...")
        
        image_ids = sorted(list(self.coco.imgs.keys()))
        if max_images:
            image_ids = image_ids[:max_images]
        
        for img_id in image_ids:
            # Get image info
            img_info = self.coco.imgs[img_id]
            image_path = self.images_path / img_info['file_name']
            
            # Skip if image file doesn't exist
            if not image_path.exists():
                continue
            
            # Get all captions for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
            
            # Create output
            result = {
                'image_path': str(image_path),
                'captions': captions
            }
            
            if output_format == 'json':
                yield json.dumps(result)
            else:
                yield result
    
    def get_sample_data(self, 
                        n_samples:int=10)-> List[Dict]:
        return self.extract_captions(max_images=n_samples)
    
    def get_all_filepaths(self)-> List[str]:
        return sorted([str(self.images_path / img_info['file_name']) for img_info in self.coco.imgs.values()])
    
    def get_all_captions(self
                         )-> Dict[Path, List[str]]:
        """
        Get all captions in the dataset
        
        Returns:
            Dict of all captions for each filepath
        """
        return {
            self.coco.imgs[img_id]['file_name']: [ann['caption'] 
                                                  for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))]
            for img_id in self.coco.imgs.keys()
        }
    
    def get_captions_for_image(self, 
                               image_name:Path
                               ) -> List[str]:
        """
        Get captions for a specific image
        
        Args:
            image_name: Name of the image file
        
        Returns:
            List of captions for the image
        """
        target_file_name = Path(image_name).name
        found_img_id = None
        
        # Iterate through the images loaded by COCO to find the matching file_name
        for img_id_candidate, img_info in self.coco.imgs.items():
            if img_info['file_name'] == target_file_name:
                found_img_id = img_id_candidate
                break
        
        if found_img_id is None:
            # If the image file_name is not found in the annotations
            return []
        
        # Get annotation IDs for the found image ID.
        # self.coco.getAnnIds expects a list of image IDs.
        ann_ids = self.coco.getAnnIds(imgIds=[found_img_id])
        anns = self.coco.loadAnns(ann_ids)
        
        return [ann['caption'] for ann in anns[:5]]
    

def main():
    ...
    
if __name__ == "__main__":
    main()