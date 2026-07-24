
import copy
import json
import os
import yaml
from PIL import Image

from PyQt6.QtWidgets import QMessageBox

from ..core.keypoint_schema import sanitize_schema
from ..utils import clamp_bbox, clamp_segmentation, keypoint_instance_bbox

from ..core.logging_config import get_logger

logger = get_logger(__name__)


def import_coco_json(file_path, class_mapping):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Validate required fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                raise ValueError(f"Missing required field '{field}' in JSON file")

        imported_annotations = {}
        image_info = {}

        # Create reverse mapping of category IDs to names
        category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # Recover per-class keypoint schemas from categories carrying a COCO
        # "keypoints" (names) field. "skeleton" is 1-based per spec, converted
        # back to the app's 0-based indices; "flip_idx" is our own export
        # extension (no COCO precedent), already 0-based. (issue #35 PR-2)
        keypoint_schemas = {}
        for cat in coco_data['categories']:
            names = cat.get('keypoints')
            if not names:
                continue
            skeleton_0based = []
            for edge in (cat.get('skeleton') or []):
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    try:
                        skeleton_0based.append([int(edge[0]) - 1, int(edge[1]) - 1])
                    except (TypeError, ValueError):
                        continue
            schema = sanitize_schema({
                "names": names,
                "skeleton": skeleton_0based,
                "flip_idx": cat.get('flip_idx'),
            })
            if schema is not None:
                keypoint_schemas[cat['name']] = schema
            else:
                logger.warning(f"Skipped malformed keypoint schema for COCO category '{cat.get('name')}'")

        # Determine the image directory
        json_dir = os.path.dirname(file_path)
        images_dir = os.path.join(json_dir, 'images')
        
        if not os.path.exists(images_dir):
            logger.warning(f"'images' subdirectory not found at {images_dir}")

        # Process images
        for image in coco_data['images']:
            try:
                file_name = image['file_name']
                image_path = os.path.join(images_dir, file_name)
                
                image_info[image['id']] = {
                    'file_name': file_name,
                    'width': int(image['width']),  # Ensure integers
                    'height': int(image['height']),
                    'path': image_path,
                    'id': int(image['id'])
                }
            except KeyError:
                logger.exception("Missing required field in image data")
                continue

        # Process annotations
        masks_dropped_for_keypoints = 0
        for ann in coco_data['annotations']:
            try:
                image_id = int(ann['image_id'])
                if image_id not in image_info:
                    logger.warning(f"Annotation refers to non-existent image ID: {image_id}")
                    continue

                if ann['category_id'] not in category_id_to_name:
                    logger.warning(f"Invalid category ID: {ann['category_id']}")
                    continue

                file_name = image_info[image_id]['file_name']
                category_name = category_id_to_name[ann['category_id']]

                if file_name not in imported_annotations:
                    imported_annotations[file_name] = {}

                if category_name not in imported_annotations[file_name]:
                    imported_annotations[file_name][category_name] = []

                annotation = {
                    'category_id': int(ann['category_id']),
                    'category_name': category_name
                }

                # Keypoint / pose instance (issue #35 PR-2) — checked before
                # segmentation/bbox handling, and skips the bbox->polygon
                # synthesis below entirely (a pose instance has no mask).
                raw_kps = ann.get('keypoints')
                if raw_kps:
                    flat = [float(v) for v in raw_kps]
                    if flat and len(flat) % 3 == 0:
                        if ann.get('segmentation'):
                            # The app's pose instance model has no mask (ADR-029)
                            # -- a source annotation carrying both is not an
                            # error, but the mask is a silent data reduction
                            # worth surfacing (e.g. real person_keypoints_*.json
                            # files often carry both).
                            masks_dropped_for_keypoints += 1
                        annotation['keypoints'] = flat
                        annotation['num_keypoints'] = int(ann.get(
                            'num_keypoints',
                            sum(1 for i in range(2, len(flat), 3) if flat[i] > 0),
                        ))
                        raw_bbox = ann.get('bbox')
                        if raw_bbox and len(raw_bbox) == 4:
                            annotation['bbox'] = [float(v) for v in raw_bbox]
                        else:
                            width = image_info[image_id]['width']
                            height = image_info[image_id]['height']
                            annotation['bbox'] = keypoint_instance_bbox(flat, width, height)
                        imported_annotations[file_name][category_name].append(annotation)
                        continue

                # Handle segmentation data
                has_valid_segmentation = False
                if 'segmentation' in ann and ann['segmentation']:  # Check if segmentation exists and is not empty
                    seg_data = ann['segmentation']
                    if isinstance(seg_data, list):
                        if seg_data and isinstance(seg_data[0], list):
                            # Take the first polygon if multiple are present
                            annotation['segmentation'] = [float(x) for x in seg_data[0]]
                            has_valid_segmentation = True
                        elif seg_data:  # Single polygon
                            annotation['segmentation'] = [float(x) for x in seg_data]
                            has_valid_segmentation = True

                # If no valid segmentation but bbox exists, create segmentation from bbox
                if not has_valid_segmentation and 'bbox' in ann:
                    x, y, w, h = [float(x) for x in ann['bbox']]
                    # Create rectangle polygon from bbox [x,y, x+w,y, x+w,y+h, x,y+h]
                    annotation['segmentation'] = [x, y, x + w, y, x + w, y + h, x, y + h]
                    annotation['type'] = 'polygon'
                    # Also store bbox data
                    annotation['bbox'] = [x, y, w, h]
                elif has_valid_segmentation:
                    annotation['type'] = 'polygon'
                elif 'bbox' in ann:  # Fallback to pure bbox if no segmentation could be created
                    annotation['bbox'] = [float(x) for x in ann['bbox']]
                    annotation['type'] = 'rectangle'

                imported_annotations[file_name][category_name].append(annotation)
                
            except (KeyError, ValueError, TypeError):
                logger.exception("Error processing annotation")
                continue

        if masks_dropped_for_keypoints:
            logger.info(
                f"{masks_dropped_for_keypoints} annotation(s) carried both "
                f"'keypoints' and a 'segmentation' -- imported as keypoints-only, "
                f"source mask(s) dropped (issue #35 PR-2)."
            )

        return imported_annotations, image_info, keypoint_schemas

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except Exception as e:
        raise ValueError(f"Error importing COCO JSON: {e}")


def import_yolo_v4(yaml_file_path, class_mapping):
    if not os.path.exists(yaml_file_path):
        raise ValueError("The selected YAML file does not exist.")
    
    directory_path = os.path.dirname(yaml_file_path)
    
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    class_names = yaml_data.get('names', [])
    if not class_names:
        raise ValueError("No class names found in the YAML file.")
    
    train_dir = os.path.join(directory_path, 'train')
    if not os.path.exists(train_dir):
        raise ValueError("No 'train' subdirectory found in the YAML file's directory.")
    
    imported_annotations = {}
    image_info = {}
    
    images_dir = os.path.join(train_dir, 'images')
    labels_dir = os.path.join(train_dir, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError("The 'train' directory must contain both 'images' and 'labels' subdirectories.")
    
    missing_images = []
    missing_labels = []
    
    for label_file in os.listdir(labels_dir):
        if label_file.lower().endswith('.txt'):
            base_name = os.path.splitext(label_file)[0]
            img_file = None
            img_path = None
            
            # Check for various image formats
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                potential_img_file = base_name + ext
                potential_img_path = os.path.join(images_dir, potential_img_file)
                if os.path.exists(potential_img_path):
                    img_file = potential_img_file
                    img_path = potential_img_path
                    break
            
            if img_path is None:
                missing_images.append(base_name)
                continue
            
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            image_id = len(image_info) + 1
            image_info[image_id] = {
                'file_name': img_file,
                'width': img_width,
                'height': img_height,
                'id': image_id,
                'path': img_path
            }
            
            imported_annotations[img_file] = {}
            
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id >= len(class_names):
                        logger.warning(f"Class ID {class_id} in {label_file} is out of range. Skipping this annotation.")
                        continue
                    class_name = class_names[class_id]
                    
                    if class_name not in imported_annotations[img_file]:
                        imported_annotations[img_file][class_name] = []
                    
                    if len(parts) == 5:  # bounding box format
                        x_center, y_center, width, height = map(float, parts[1:5])
                        x1 = (x_center - width/2) * img_width
                        y1 = (y_center - height/2) * img_height
                        x2 = (x_center + width/2) * img_width
                        y2 = (y_center + height/2) * img_height
                        
                        annotation = {
                            'category_id': class_id,
                            'category_name': class_name,
                            'type': 'rectangle',
                            'bbox': [x1, y1, x2-x1, y2-y1]
                        }
                    else:  # polygon format
                        polygon = [float(coord) * (img_width if i % 2 == 0 else img_height) for i, coord in enumerate(parts[1:])]
                        
                        annotation = {
                            'category_id': class_id,
                            'category_name': class_name,
                            'type': 'polygon',
                            'segmentation': polygon
                        }
                    
                    imported_annotations[img_file][class_name].append(annotation)
    
    # Check for images without labels
    for img_file in os.listdir(images_dir):
        base_name, ext = os.path.splitext(img_file)
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
            label_file = base_name + '.txt'
            if not os.path.exists(os.path.join(labels_dir, label_file)):
                missing_labels.append(img_file)
    
    if missing_images or missing_labels:
        message = "The following issues were found:\n\n"
        if missing_images:
            message += f"Labels without corresponding images: {', '.join(missing_images)}\n\n"
        if missing_labels:
            message += f"Images without corresponding labels: {', '.join(missing_labels)}\n\n"
        message += "Do you want to continue importing the remaining data?"
        
        reply = QMessageBox.question(None, "Import Issues", message, 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.No:
            raise ValueError("Import cancelled due to missing files.")

    # Legacy format stays detection-only (issue #35 PR-2) — no keypoint
    # schemas to recover, but the 3-tuple contract must stay uniform across
    # every import_* entry point.
    return imported_annotations, image_info, {}


def import_yolo_v5plus(yaml_file_path, class_mapping):
    """
    Import annotations from YOLO v5+ format.
    Expected directory structure:
    root_dir/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """
    if not os.path.exists(yaml_file_path):
        raise ValueError("The selected YAML file does not exist.")
    
    root_dir = os.path.dirname(yaml_file_path)
    
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    class_names = yaml_data.get('names', [])
    if not class_names:
        raise ValueError("No class names found in the YAML file.")

    # YOLO-pose declares one dataset-global kpt_shape/flip_idx (issue #35
    # PR-2) — not one per class — so every class in `names` is treated as a
    # pose class with this K, even one with zero instances in this label set.
    kpt_shape = yaml_data.get('kpt_shape')
    pose_k = None
    if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
        try:
            pose_k = int(kpt_shape[0]) or None
        except (TypeError, ValueError):
            pose_k = None

    imported_annotations = {}
    image_info = {}

    # Process both train and val directories
    for split in ['train', 'val']:
        images_dir = os.path.join(root_dir, 'images', split)
        labels_dir = os.path.join(root_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            logger.warning(f"{split} directory not found, skipping")
            continue
        
        for label_file in os.listdir(labels_dir):
            if label_file.lower().endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                img_file = None
                img_path = None
                
                # Check for various image formats
                for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                    potential_img_file = base_name + ext
                    potential_img_path = os.path.join(images_dir, potential_img_file)
                    if os.path.exists(potential_img_path):
                        img_file = potential_img_file
                        img_path = potential_img_path
                        break
                
                if img_path is None:
                    logger.warning(f"No image found for label {label_file}")
                    continue
                
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                image_id = len(image_info) + 1
                image_info[image_id] = {
                    'file_name': img_file,
                    'width': img_width,
                    'height': img_height,
                    'id': image_id,
                    'path': img_path
                }
                
                imported_annotations[img_file] = {}
                
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id >= len(class_names):
                            logger.warning(f"Class ID {class_id} in {label_file} is out of range")
                            continue
                        class_name = class_names[class_id]
                        
                        if class_name not in imported_annotations[img_file]:
                            imported_annotations[img_file][class_name] = []

                        # Disambiguated purely by token count: kpt_shape in
                        # data.yaml declares this WHOLE dataset pose-only (issue
                        # #35 PR-2), so a line with 5+3*pose_k tokens is always
                        # a pose instance, never a same-length segmentation
                        # polygon -- YOLO-pose datasets don't mix in polygons.
                        if pose_k and len(parts) == 5 + 3 * pose_k:  # YOLO-pose format
                            x_center, y_center, width, height = map(float, parts[1:5])
                            x1 = (x_center - width/2) * img_width
                            y1 = (y_center - height/2) * img_height
                            w = width * img_width
                            h = height * img_height

                            flat = []
                            for i in range(5, len(parts), 3):
                                flat.extend([
                                    float(parts[i]) * img_width,
                                    float(parts[i + 1]) * img_height,
                                    float(parts[i + 2]),
                                ])

                            annotation = {
                                'category_id': class_id,
                                'category_name': class_name,
                                'keypoints': flat,
                                'num_keypoints': sum(1 for i in range(2, len(flat), 3) if flat[i] > 0),
                                'bbox': [x1, y1, w, h],
                            }
                        elif len(parts) == 5:  # bounding box format
                            x_center, y_center, width, height = map(float, parts[1:5])
                            x1 = (x_center - width/2) * img_width
                            y1 = (y_center - height/2) * img_height
                            w = width * img_width
                            h = height * img_height

                            annotation = {
                                'category_id': class_id,
                                'category_name': class_name,
                                'type': 'rectangle',
                                'bbox': [x1, y1, w, h]
                            }
                        else:  # polygon format
                            polygon = []
                            for i in range(1, len(parts), 2):
                                x = float(parts[i]) * img_width
                                y = float(parts[i+1]) * img_height
                                polygon.extend([x, y])
                            
                            annotation = {
                                'category_id': class_id,
                                'category_name': class_name,
                                'type': 'polygon',
                                'segmentation': polygon
                            }
                        
                        imported_annotations[img_file][class_name].append(annotation)

    # Applied uniformly to every declared class (see kpt_shape comment above),
    # not just classes observed with pose-shaped lines. Generic kp0..kp{K-1}
    # names — YOLO-pose carries no point names. copy.deepcopy per class so no
    # two class entries alias the same schema dict. (issue #35 PR-2)
    keypoint_schemas = {}
    if pose_k:
        schema = sanitize_schema({
            "names": [f"kp{i}" for i in range(pose_k)],
            "skeleton": [],
            "flip_idx": yaml_data.get('flip_idx'),
        })
        if schema is not None:
            for name in class_names:
                keypoint_schemas[name] = copy.deepcopy(schema)

    return imported_annotations, image_info, keypoint_schemas



def _voc_mask_polygons(mask_path):
    """Polygons per colour region in a VOC segmentation mask PNG.

    Returns ``{(r, g, b): [flat polygon, ...]}``. ``export_pascal_voc_both``
    writes one colour per class, so the colour is the only link back from a
    mask region to its class name — which is why the caller pairs these with
    the XML objects rather than trusting order.

    Contour extraction mirrors ``inference/sam_utils._mask_to_polygon``: same
    ``RETR_EXTERNAL`` / ``CHAIN_APPROX_SIMPLE`` pass and the same >= 6
    coordinate floor, so a mask round-tripped through export and back produces
    the same kind of outline the app makes natively.
    """
    import cv2
    import numpy as np

    image = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if image is None:
        return {}
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    polygons = {}
    for colour in np.unique(rgb.reshape(-1, 3), axis=0):
        if not colour.any():
            continue  # pure black is the background
        binary = np.all(rgb == colour, axis=-1).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        found = []
        for contour in contours:
            if cv2.contourArea(contour) <= 10:
                continue
            flat = contour.flatten().tolist()
            if len(flat) >= 6:
                found.append([float(c) for c in flat])
        if found:
            polygons[tuple(int(c) for c in colour)] = found
    return polygons


def import_pascal_voc(directory_path, class_mapping):
    """Import a directory of Pascal VOC XML annotations (issue #75).

    Closes a plain asymmetry in ``io/``: the app has exported VOC since before
    this change but could never read its own output back, let alone the large
    amount of VOC-format data in the wild.

    ``directory_path`` may be the dataset root (containing ``Annotations/`` and
    ``images/``, the layout ``export_pascal_voc_bbox`` writes) or the
    ``Annotations`` directory itself — both are what a user actually picks.

    Where ``SegmentationClass`` mask PNGs accompany the XML (as
    ``export_pascal_voc_both`` writes), polygons are reconstructed so a full
    round-trip is possible.

    Returns the uniform ``(annotations, image_info, keypoint_schemas)`` triple
    every entry point in this module returns. The schema dict is always empty:
    VOC has no keypoint concept. Breaking that shape would break the caller.
    """
    import xml.etree.ElementTree as ET

    if not os.path.isdir(directory_path):
        raise ValueError("The selected Pascal VOC path is not a directory.")

    annotations_dir = directory_path
    if os.path.isdir(os.path.join(directory_path, "Annotations")):
        annotations_dir = os.path.join(directory_path, "Annotations")
    root_dir = os.path.dirname(annotations_dir.rstrip(os.sep)) or directory_path

    xml_files = sorted(
        name for name in os.listdir(annotations_dir) if name.lower().endswith(".xml")
    )
    if not xml_files:
        raise ValueError(
            f"No .xml annotation files found in {annotations_dir}."
        )

    masks_dir = None
    for candidate in ("SegmentationClass", "SegmentationObject", "masks"):
        path = os.path.join(root_dir, candidate)
        if os.path.isdir(path):
            masks_dir = path
            break

    imported_annotations = {}
    image_info = {}
    next_class_id = max(class_mapping.values(), default=0) + 1
    local_mapping = dict(class_mapping)

    for xml_name in xml_files:
        xml_path = os.path.join(annotations_dir, xml_name)
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            # Abort rather than import half a dataset: a partially-imported
            # project is harder to recover from than a refused import.
            raise ValueError(f"Malformed Pascal VOC XML in {xml_name}: {exc}")
        root = tree.getroot()

        file_name = (root.findtext("filename") or "").strip()
        if not file_name:
            file_name = os.path.splitext(xml_name)[0] + ".png"

        size = root.find("size")
        img_width = int(float(size.findtext("width", "0"))) if size is not None else 0
        img_height = int(float(size.findtext("height", "0"))) if size is not None else 0

        image_info[file_name] = {
            "file_name": file_name,
            "width": img_width,
            "height": img_height,
            "id": len(image_info) + 1,
        }
        imported_annotations.setdefault(file_name, {})

        mask_polygons = {}
        if masks_dir:
            stem = os.path.splitext(file_name)[0]
            for extension in (".png", ".PNG"):
                mask_path = os.path.join(masks_dir, stem + extension)
                if os.path.exists(mask_path):
                    mask_polygons = _voc_mask_polygons(mask_path)
                    break

        # Mask polygons are consumed per class in XML order, so two objects of
        # the same class each take one region rather than both taking the first.
        polygon_queue = {}

        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            if not class_name:
                continue
            if class_name not in local_mapping:
                local_mapping[class_name] = next_class_id
                next_class_id += 1
            class_id = local_mapping[class_name]

            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
            except (TypeError, ValueError):
                logger.warning(
                    "skipping an object with unreadable bndbox in %s", xml_name
                )
                continue

            # VOC stores corners; the app stores [x, y, width, height].
            # Convert, never copy.
            bbox = [xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin)]
            # Producers disagree on whether VOC coordinates are 0- or 1-based,
            # so clamp into the image rather than trusting the file (ADR-024).
            if img_width > 0 and img_height > 0:
                bbox = clamp_bbox(bbox, img_width, img_height)

            annotation = {
                "category_id": class_id,
                "category_name": class_name,
                "type": "rectangle",
                "bbox": bbox,
            }

            # `difficult` and `truncated` have no home in the data model.
            # Ignored deliberately rather than invented into new fields.

            if mask_polygons:
                if class_name not in polygon_queue:
                    polygon_queue[class_name] = _polygons_for_class(
                        mask_polygons, class_name, local_mapping
                    )
                queue = polygon_queue[class_name]
                if queue:
                    polygon = queue.pop(0)
                    if img_width > 0 and img_height > 0:
                        polygon = clamp_segmentation(polygon, img_width, img_height)
                    annotation["segmentation"] = polygon
                    annotation["type"] = "polygon"

            imported_annotations[file_name].setdefault(class_name, []).append(annotation)

    # VOC has no keypoint concept, but the triple's shape is the contract.
    return imported_annotations, image_info, {}


def _polygons_for_class(mask_polygons, class_name, class_mapping):
    """Mask polygons belonging to ``class_name``, best-effort.

    ``export_pascal_voc_both`` paints each class in a colour derived from its
    id, so the id indexes the colour list. When the colour cannot be resolved
    (a mask from a foreign producer with its own palette) every unclaimed
    region is offered instead — a polygon on the right object with a slightly
    uncertain provenance beats no polygon at all, and the bbox is authoritative
    either way.
    """
    class_id = class_mapping.get(class_name)
    if class_id is not None:
        for colour, polygons in mask_polygons.items():
            if colour[0] == class_id or colour[1] == class_id or colour[2] == class_id:
                return list(polygons)
    return [polygon for polygons in mask_polygons.values() for polygon in polygons]


def process_import_format(import_format, file_path, class_mapping):
    if import_format == "COCO JSON":
        return import_coco_json(file_path, class_mapping)
    elif import_format == "YOLO (v4 and earlier)":
        return import_yolo_v4(file_path, class_mapping)  # Still using same function, just updated format name
    elif import_format == "YOLO (v5+)":
        return import_yolo_v5plus(file_path, class_mapping)  # New format handling
    elif import_format == "Pascal VOC":
        return import_pascal_voc(file_path, class_mapping)  # issue #75
    else:
        raise ValueError(f"Unsupported import format: {import_format}")


