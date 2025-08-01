#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import base64
import numpy as np
import torch
import random
import signal
import sys
import matplotlib.pyplot as plt

def encode_image(image_path):
    """
    Encode an image to base64
    """
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def get_device():
    """
    Get available device (CUDA or CPU)
    """
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimal_batch_size(initial_size=8):
    """
    Estimate optimal batch size based on available GPU memory
    """
    if not torch.cuda.is_available():
        return initial_size
    
    try:
        # Get GPU memory information
        free_memory, total_memory = torch.cuda.mem_get_info(0)
        free_gb = free_memory / (1024**3)
        
        # Adjust batch size based on available memory
        if free_gb > 20:  # More than 20GB free
            return 16
        elif free_gb > 10:  # More than 10GB free
            return 8
        elif free_gb > 5:  # More than 5GB free
            return 4
        else:
            return 2
    except Exception:
        # Return initial value if memory info retrieval fails
        return initial_size

def setup_japanese_font():
    """
    Set up fonts for plotting
    Use English display instead of Japanese
    """
    import matplotlib
    
    # Set English font
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    print("Plot display set to English")

def signal_handler(sig, frame):
    """
    Signal handler function: Process interrupt signals like Ctrl+C
    """
    print("\nInterrupting process. Exiting safely...")
    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

def get_optimized_concept_list():
    """Return concept list optimized for rare class detection"""
    return [
        # Construction vehicles (detailed)
        "crane", "bulldozer", "excavator", "forklift", "cement_mixer", "road_roller",
        "backhoe", "cherry_picker", "construction_vehicle", "dump_truck", "concrete_pump",
        "asphalt_paver", "compactor", "grader", "trencher", "pile_driver",
        
        # Bicycle/motorcycle related (detailed)
        "bicycle", "mountain_bike", "road_bike", "delivery_bike", "electric_bike",
        "motorcycle", "scooter", "moped", "dirt_bike", "sport_motorcycle", "cruiser_motorcycle",
        
        # Trailers and special vehicles
        "trailer", "semi_trailer", "flatbed_trailer", "car_carrier_trailer", "livestock_trailer",
        "boat_trailer", "camper_trailer", "horse_trailer", "logging_trailer", "tanker_trailer",
        
        # Truck related
        "truck", "pickup_truck", "delivery_truck", "tow_truck", "garbage_truck", "fire_truck",
        "tank_truck", "box_truck", "utility_truck", "monster_truck", "snow_plow_truck",
        
        # Common classes (include only a few)
        "car", "pedestrian", "traffic_light", "traffic_sign",
        
        # Special situations
        "oversize_load", "wide_load", "unusual_cargo", "construction_site", "road_work",
        "accident_scene", "broken_down_vehicle", "vehicle_on_fire", "flooded_road",
        "fallen_tree_on_road", "landslide", "sinkhole", "road_debris"
    ]

def is_potential_rare_class(class_name):
    """Quick check if a class name could be a rare class"""
    rare_keywords = ["construction", "bicycle", "motorcycle", "trailer", "truck", 
                    "bulldozer", "excavator", "forklift", "cement", "roller"]
    return any(keyword in class_name.lower() for keyword in rare_keywords)

def map_to_target_class(concept_name, target_classes, similarity_threshold=0.7):
    """
    Map concept name to a similar rare class name
    Example: 'bulldozer' → 'construction_vehicle'
    """
    # Rare class mapping dictionary - enhanced keywords
    class_mappings = {
        "construction_vehicle": [
            # Construction vehicle types
            "bulldozer", "excavator", "forklift", "cement_mixer", "road_roller",
            "backhoe", "cherry_picker", "dump_truck", "concrete_pump", "crane",
            "construction", "digger", "loader", "grader", "paver", "trencher",
            "scraper", "compactor", "pile_driver", "concrete_mixer", "steamroller",
            # Construction-related truck terms
            "dump", "construction_truck", "concrete", "heavy_machinery", "earth_mover",
            "construction_equipment", "heavy_equipment", "construction_site_truck"
        ],
        "bicycle": [
            "bike", "mountain_bike", "road_bike", "delivery_bike", "electric_bike",
            "cyclist", "biking", "cycling"
        ],
        "motorcycle": [
            "scooter", "moped", "dirt_bike", "sport_motorcycle", "cruiser_motorcycle",
            "motorbike", "biker"
        ],
        "trailer": [
            "semi_trailer", "flatbed_trailer", "car_carrier_trailer", "livestock_trailer",
            "boat_trailer", "camper_trailer", "horse_trailer", "logging_trailer", "tanker_trailer"
        ],
        "truck": [
            "pickup_truck", "delivery_truck", "tow_truck", "garbage_truck", "fire_truck",
            "tank_truck", "box_truck", "utility_truck", "monster_truck", "snow_plow_truck"
        ]
    }
    
    # Convert input concept name to lowercase
    concept_lower = concept_name.lower()
    
    # Direct match with target classes
    if concept_lower in [tc.lower() for tc in target_classes]:
        return concept_name
    
    # Special case: Distinguishing between "truck" and "construction_vehicle"
    # When YOLO detects "truck", evaluate if it might be a construction vehicle
    if concept_lower == "truck" and "construction_vehicle" in target_classes:
        # High priority keywords indicating construction-related trucks
        construction_truck_keywords = ["dump", "mixer", "concrete", "construction", "heavy", 
                                      "crane", "excavator", "bulldozer", "loader", "earth"]
        
        # Consider target_classes order
        # If construction_vehicle has higher priority (appears earlier in the list)
        # increase the likelihood of mapping trucks to construction vehicles
        if "construction_vehicle" in target_classes and "truck" in target_classes:
            cv_index = target_classes.index("construction_vehicle")
            truck_index = target_classes.index("truck")
            
            if cv_index < truck_index:
                # Higher priority for construction_vehicle, always treat as construction vehicle
                return "construction_vehicle"
        
        # Check if the truck should be treated as a construction vehicle
        if any(keyword in concept_name.lower() for keyword in construction_truck_keywords):
            return "construction_vehicle"
        else:
            return "truck"
    
    # Check for keyword matches with each target class
    for target_class, keywords in class_mappings.items():
        if target_class not in target_classes:
            continue
        
        # Check if any keyword is contained in the concept
        for keyword in keywords:
            if keyword.lower() in concept_lower:
                return target_class
    
    # Return original name if no match found
    return concept_name

def map_class_name_to_nusc(class_name):
    """
    Map YOLO class names to nuScenes class names
    Especially used when skipping caption generation

    Args:
        class_name: Class name detected by YOLO or similar
    
    Returns:
        Corresponding nuScenes class name(s). Returns list if multiple candidates.
    """
    # Convert class name to lowercase
    class_name_lower = class_name.lower()
    
    # Mapping dictionary
    mapping = {
        # Vehicle related
        "truck": ["truck", "construction_vehicle", "trailer"],
        "pickup": ["truck"],
        "lorry": ["truck"],
        "car": ["car"],
        "auto": ["car"],
        "automobile": ["car"],
        "vehicle": ["car"],
        "sedan": ["car"],
        "suv": ["car"],
        "van": ["car"],
        "taxi": ["car"],
        "bus": ["bus"],
        "coach": ["bus"],
        "trailer": ["trailer"],
        "semi": ["trailer", "truck"],
        "construction": ["construction_vehicle"],
        "bulldozer": ["construction_vehicle"],
        "excavator": ["construction_vehicle"],
        "crane": ["construction_vehicle"],
        "forklift": ["construction_vehicle"],
        
        # Bicycle/motorcycle related
        "bicycle": ["bicycle"],
        "bike": ["bicycle"],
        "cycling": ["bicycle"],
        "motorcycle": ["motorcycle"],
        "motorbike": ["motorcycle"],
        "scooter": ["motorcycle"],
        
        # Pedestrian related
        "person": ["pedestrian"],
        "pedestrian": ["pedestrian"],
        "human": ["pedestrian"],
        
        # Traffic related
        "traffic light": ["traffic_light"],
        "traffic_light": ["traffic_light"],
        "stoplight": ["traffic_light"],
        "traffic sign": ["traffic_sign"],
        "traffic_sign": ["traffic_sign"],
        "roadsign": ["traffic_sign"],
        "street sign": ["traffic_sign"],
        "stop sign": ["traffic_sign"],
        "yield sign": ["traffic_sign"],
        "speed limit": ["traffic_sign"],
        "barrier": ["barrier"],
        "barricade": ["barrier"],
        "fence": ["barrier"],
        "guardrail": ["barrier"],
        "roadblock": ["barrier"],
        "traffic cone": ["traffic_cone"],
        "traffic_cone": ["traffic_cone"],
        "cone": ["traffic_cone"],
        "pylon": ["traffic_cone"],
    }
    
    # Search for keywords matching the class name
    matched_classes = []
    for keyword, classes in mapping.items():
        if keyword in class_name_lower:
            matched_classes.extend(classes)
    
    # Remove duplicates
    matched_classes = list(set(matched_classes))
    
    # Return matches if found, otherwise return the original class name
    if matched_classes:
        return matched_classes
    else:
        return [class_name] 