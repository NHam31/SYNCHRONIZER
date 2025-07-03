
Cette section détaille l'architecture, le fonctionnement et l'implémentation de chaque modèle utilisé dans le pipeline Segma Vision Pro HLBB.

SAM (Segment Anything Model)
============================

Vue d'ensemble
--------------

SAM est un modèle de segmentation révolutionnaire développé par Meta AI qui peut segmenter n'importe quel objet dans une image sans formation préalable sur des classes spécifiques.
SAM a acquis une notion générale de ce que sont les objets, cette compréhension lui permet de généraliser en zéro-shot à des objets et des images inconnus sans nécessiter d’entraînement supplémentaire. Il peut segmenter les objets de différentes manières, notamment à l’aide d’une boîte englobante (bounding box), d’un clic (curseur/pointeur), d’un masque existant ou d’un texte descriptif

Architecture
------------

**Encodeur d'Image (Vision Transformer)**
    * Architecture : ViT-H (Vision Transformer Huge)
    * Paramètres : 630M paramètres
    * Résolution d'entrée : 1024×1024 pixels
    * Embedding dimension : 1280
    * Nombre de layers : 32
    * Attention heads : 16

**Encodeur de Prompt**
    * Points : Coordonnées (x,y) avec labels positif/négatif
    * Boîtes : Bounding boxes [x1,y1,x2,y2]
    * Masques : Masques de segmentation low-resolution
    * Texte : Pas directement supporté (via autres modèles)

**Décodeur de Masque**
    * Architecture : Transformer léger
    * Output : Masques haute résolution (1024×1024)
    * Multiples hypothèses : 3 masques par prompt avec scores de qualité

Fonctionnement
--------------

**Étape 1 : Preprocessing**

.. code-block:: python

   # Redimensionnement et normalisation
   image = resize_image(image, target_size=(1024, 1024))
   image_tensor = normalize(image, mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])

**Étape 2 : Encoding d'Image**

.. code-block:: python

   # Extraction des features avec ViT
   image_embeddings = vit_encoder(image_tensor)  # Shape: [1, 256, 64, 64]

**Étape 3 : Génération Automatique de Prompts**

.. code-block:: python

   # Grille de points pour segmentation automatique
   points_grid = generate_point_grid(image_size=(1024, 1024), 
                                   points_per_side=32)
   # Résultat : 1024 points uniformément distribués

**Étape 4 : Génération de Masques**

.. code-block:: python

   # Pour chaque point de la grille
   for point in points_grid:
       masks, scores, logits = sam_predictor.predict(
           point_coords=point,
           point_labels=[1],  # Point positif
           multimask_output=True
       )

Input/Output
------------

**Input**
    * Image RGB : Dimensions variables (redimensionnée vers 1024×1024)
    * Prompts optionnels : Points, boîtes, ou masques existants

**Output**
    * Masques binaires : Résolution native de l'image
    * Scores de qualité : Float [0,1] pour chaque masque
    * Logits : Valeurs brutes avant sigmoid pour post-processing

Avantages
---------

* **Zero-shot** : Pas besoin d'entraînement sur des classes spécifiques
* **Haute qualité** : Segmentation précise même sur objets complexes
* **Rapidité** : Inference en temps réel après encoding initial
* **Flexibilité** : Accepte différents types de prompts

Limitations
-----------

* **Mémoire** : Requiert 8GB+ VRAM pour ViT-H
* **Pas de sémantique** : Ne comprend pas ce qu'il segmente
* **Bruit** : Peut créer des masques sur du bruit ou arrière-plan

BLIP (Bootstrapping Language-Image Pre-training)
================================================

Vue d'ensemble
--------------

BLIP est un modèle multimodal qui comprend et génère du texte à partir d'images. Il excelle dans les tâches de description d'images (captioning) et de compréhension visuelle.

Architecture
------------

**Vision Encoder**
    * Base : ViT-B/16 (Vision Transformer Base)
    * Paramètres : 86M paramètres
    * Résolution : 384×384 pixels
    * Patch size : 16×16
    * Embedding dimension : 768

**Text Encoder (BERT-like)**
    * Architecture : Transformer encoder
    * Vocabulaire : 30,522 tokens
    * Max sequence length : 512 tokens
    * Hidden size : 768
    * Layers : 12
    * Attention heads : 12

**Text Decoder (GPT-like)**
    * Architecture : Causal language model
    * Paramètres : 84M paramètres
    * Génération autoregressive
    * Beam search pour optimisation

**Multimodal Fusion**
    * Cross-attention entre vision et texte
    * Shared attention layers
    * Task-specific heads pour différentes applications

Fonctionnement
--------------

**Étape 1 : Vision Encoding**

.. code-block:: python

   # Preprocessing de l'image
   image = preprocess_image(roi_image)  # 384x384
   vision_features = vision_encoder(image)  # [197, 768]
   # 197 = 1 [CLS] + 196 patches (14x14)

**Étape 2 : Text Generation (Captioning)**

.. code-block:: python

   # Génération de description
   caption = blip_model.generate(
       image=image,
       sample=False,  # Deterministic
       num_beams=3,   # Beam search
       max_length=50,
       min_length=10
   )

**Étape 3 : Cross-Modal Understanding**

.. code-block:: python

   # Fusion vision-texte pour compréhension
   multimodal_features = cross_attention(
       vision_features, 
       text_features
   )

Preprocessing
-------------

**Images**
    * Resize vers 384×384
    * Normalisation : mean=[0.48145466, 0.4578275, 0.40821073]
    * Standard deviation : [0.26862954, 0.26130258, 0.27577711]

**Texte**
    * Tokenisation avec BERT tokenizer
    * Padding/Truncation vers longueur fixe
    * Ajout de tokens spéciaux [CLS], [SEP]

Input/Output
------------

**Input**
    * Image : RGB, résolution variable → 384×384
    * Texte optionnel : Pour tasks de compréhension

**Output (Captioning)**
    * Description textuelle : 10-50 tokens
    * Score de confiance : Probabilité du sequence

**Output (VQA - Visual Question Answering)**
    * Réponse courte : Généralement 1-5 mots
    * Classification probability

Applications dans notre Pipeline
--------------------------------

**1. Description de Masques SAM**

.. code-block:: python

   # Pour chaque masque SAM
   for mask in sam_masks:
       roi = extract_roi(image, mask)
       description = blip_model.generate(roi)
       # Exemple: "a brown dog sitting on grass"

**2. Extraction de Concepts**

.. code-block:: python

   # Descriptions collectées
   descriptions = [
       "a brown dog sitting on grass",
       "a person wearing blue jacket",
       "green trees in background"
   ]
   # → Envoyé à Mistral pour extraction de classes

Mistral LLM
===========

Vue d'ensemble
--------------

Mistral est un modèle de langage développé par Mistral AI, optimisé pour la compréhension, génération et traitement de texte avec une efficacité computationnelle élevée.

Architecture
------------

**Architecture de Base**
    * Type : Transformer Decoder-only
    * Paramètres : 7B (Mistral-7B) ou 8x7B (Mixtral)
    * Context length : 32,768 tokens (32k context window)
    * Vocabulary size : 32,000 tokens
    * Hidden dimension : 4,096
    * Intermediate size : 14,336
    * Number of layers : 32
    * Attention heads : 32
    * Key-value heads : 8 (Grouped Query Attention)

**Innovations Techniques**
    * **Sliding Window Attention** : Attention locale sur 4,096 tokens
    * **Grouped Query Attention (GQA)** : Réduction mémoire
    * **SwiGLU Activation** : Fonction d'activation optimisée
    * **RMSNorm** : Normalisation efficace

Fonctionnement dans notre Pipeline
----------------------------------

**Rôle : Extraction et Structuration de Classes**

**Étape 1 : Collecte des Descriptions BLIP**

.. code-block:: python

   # Input : Descriptions de tous les masques SAM
   blip_descriptions = [
       "a brown dog with white markings sitting on green grass",
       "a person wearing a blue denim jacket and dark pants",
       "large green trees with dense foliage in the background",
       "a walking path made of concrete or stone"
   ]

**Étape 2 : Prompt Engineering**

.. code-block:: python

   prompt = f"""
   Analyze these image descriptions and extract the main object classes for object detection:

   Descriptions:
   {chr(10).join(blip_descriptions)}

   Extract only the main objects as single words or short phrases, suitable for object detection:
   - Focus on concrete, visible objects
   - Avoid adjectives and descriptions
   - Format as comma-separated list
   - Maximum 10 classes

   Classes:
   """

**Étape 3 : Génération et Parsing**

.. code-block:: python

   # Appel API Mistral
   response = mistral_client.chat(
       model="mistral-medium",
       messages=[{"role": "user", "content": prompt}],
       temperature=0.1,  # Déterministe
       max_tokens=100
   )
   
   # Parsing du résultat
   classes = response.choices[0].message.content.strip()
   class_list = [cls.strip() for cls in classes.split(',')]
   # Résultat: ["dog", "person", "tree", "path"]

**Étape 4 : Validation et Filtrage**

.. code-block:: python

   # Filtrage des classes valides
   valid_classes = []
   for cls in class_list:
       if len(cls.split()) <= 2 and len(cls) >= 3:  # Mots simples
           valid_classes.append(cls)

Input/Output
------------

**Input**
    * Descriptions textuelles de BLIP
    * Prompt structuré pour extraction de classes
    * Context : Compréhension de la tâche de détection d'objets

**Output**
    * Liste de classes/objets : Format ["class1", "class2", ...]
    * Classes nettoyées et structurées
    * Prêtes pour Grounding DINO

Avantages
---------

* **Compréhension contextuelle** : Comprend la sémantique des descriptions
* **Structuration intelligente** : Extrait automatiquement les concepts pertinents
* **Flexibilité** : S'adapte à différents types de contenu
* **Efficacité** : Traitement rapide de texte

Configuration API
-----------------

.. code-block:: python

   from mistralai.client import MistralClient

   client = MistralClient(api_key="your-api-key")
   
   # Configuration optimale pour notre use case
   config = {
       "model": "mistral-medium",
       "temperature": 0.1,      # Déterministe
       "max_tokens": 100,       # Suffisant pour liste de classes
       "top_p": 0.9            # Diversité contrôlée
   }

Grounding DINO
==============

Vue d'ensemble
--------------

Grounding DINO est un modèle révolutionnaire qui combine détection d'objets et compréhension de langage naturel, permettant de détecter des objets à partir de descriptions textuelles.

Architecture
------------

**Backbone Vision**
    * Base : Swin Transformer
    * Variants : Swin-T, Swin-B, Swin-L
    * Multi-scale feature extraction
    * Hierarchical attention

**Text Encoder**
    * Architecture : BERT-base
    * Fine-tuné pour détection
    * Cross-modal alignment
    * Context-aware embeddings

**Feature Fusion**
    * **Cross-Attention Layers** : Fusion vision-texte
    * **Multimodal Transformer** : 6 couches de fusion
    * **Position Encoding** : Spatial et textuel

**Detection Head**
    * **Classification Head** : Probabilités d'objets
    * **Regression Head** : Coordonnées de bounding boxes
    * **DETR-style** : Set prediction sans NMS

Fonctionnement Détaillé
-----------------------

**Étape 1 : Text Processing**

.. code-block:: python

   # Input : Classes extraites par Mistral
   text_queries = ["dog", "person", "tree", "path"]
   
   # Tokenisation et encoding
   text_tokens = tokenizer(text_queries, return_tensors="pt")
   text_features = text_encoder(text_tokens)  # [4, 768]

**Étape 2 : Image Feature Extraction**

.. code-block:: python

   # Multi-scale feature extraction
   image_features = swin_backbone(image)
   # Output: Liste de features maps à différentes résolutions
   # [B, C1, H1, W1], [B, C2, H2, W2], [B, C3, H3, W3], [B, C4, H4, W4]

**Étape 3 : Cross-Modal Fusion**

.. code-block:: python

   # Fusion des modalités vision et texte
   for layer in fusion_layers:
       image_features, text_features = layer(
           image_features, 
           text_features,
           cross_attention=True
       )

**Étape 4 : Object Detection**

.. code-block:: python

   # Prédiction des objets
   predictions = detection_head(fused_features)
   
   # Post-processing
   boxes, scores, labels = post_process(
       predictions,
       confidence_threshold=0.25,
       text_queries=text_queries
   )

Input/Output Détaillé
--------------------

**Input**
    * **Image** : RGB, résolution variable (redimensionnée)
    * **Text Prompts** : Liste de classes ["class1", "class2", ...]
    * **Confidence Threshold** : Seuil de détection (défaut: 0.25)

**Output**
    * **Bounding Boxes** : Coordonnées [x1, y1, x2, y2] normalisées
    * **Scores** : Probabilités de détection [0, 1]
    * **Labels** : Index correspondant aux classes input
    * **Text Alignment** : Correspondance avec prompts textuels

**Format de Sortie**

.. code-block:: python

   results = {
       'boxes': tensor([[0.1, 0.2, 0.4, 0.6],    # Boîte 1
                       [0.5, 0.3, 0.8, 0.7]]),   # Boîte 2
       'scores': tensor([0.85, 0.72]),            # Scores
       'labels': tensor([0, 1]),                  # dog=0, person=1
       'text_queries': ["dog", "person"]
   }

Preprocessing
-------------

**Image Preprocessing**

.. code-block:: python

   # Standardisation
   transform = Compose([
       Resize((800, 1333)),  # Resize intelligent
       ToTensor(),
       Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
   ])

**Text Preprocessing**

.. code-block:: python

   # Formatage des prompts
   def format_text_prompts(classes):
       # "dog. person. tree. path."
       return '. '.join(classes) + '.'

Optimisations et Configuration
-----------------------------

**Hyperparamètres Clés**

.. code-block:: python

   config = {
       'confidence_threshold': 0.25,  # Seuil de détection
       'box_threshold': 0.35,         # Seuil pour boîtes
       'text_threshold': 0.25,        # Seuil texte-image
       'device': 'cuda',              # GPU obligatoire
       'fp16': True                   # Précision mixte
   }

**Filtrage Post-Processing**

.. code-block:: python

   # Suppression des détections redondantes
   def remove_enclosing_boxes(boxes, scores, threshold=0.8):
       keep_indices = []
       for i, box in enumerate(boxes):
           is_enclosed = False
           for j, other_box in enumerate(boxes):
               if i != j and is_box_enclosed(box, other_box, threshold):
                   is_enclosed = True
                   break
           if not is_enclosed:
               keep_indices.append(i)
       return boxes[keep_indices], scores[keep_indices]

HLBB Features (61 dimensions)
=============================

Vue d'ensemble
--------------

Le système HLBB (High-Level Bounding Box) extrait 61 caractéristiques quantitatives pour chaque objet détecté, combinant analyse colorimétrique, texturale et géométrique.

Architecture des Features
-------------------------

**Répartition des 61 Dimensions**
    * **Histogramme Couleur RGB** : 48 dimensions (4×4×3)
    * **Texture LBP (Local Binary Pattern)** : 10 dimensions
    * **Caractéristiques Géométriques** : 3 dimensions

1. Histogramme Couleur RGB (48 dimensions)
------------------------------------------

**Principe**
    Analyse de la distribution des couleurs dans la région d'intérêt (ROI) en divisant l'espace colorimétrique RGB en bins discrets.

**Implémentation**

.. code-block:: python

   def extract_color_histogram(roi):
       # Calcul histogramme 3D : 4 bins par canal RGB
       hist = cv2.calcHist(
           [roi],                           # Image ROI
           [0, 1, 2],                      # Canaux R, G, B
           None,                           # Pas de masque
           [4, 4, 3],                      # 4×4×3 = 48 bins
           [0, 256, 0, 256, 0, 256]       # Ranges pour chaque canal
       )
       
       # Normalisation L2
       hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
       return hist.flatten()  # Shape: (48,)

**Interprétation des Bins**
    * **Rouge** : 4 niveaux [0-64, 64-128, 128-192, 192-255]
    * **Vert** : 4 niveaux identiques
    * **Bleu** : 3 niveaux [0-85, 85-170, 170-255]
    * **Total** : 4×4×3 = 48 combinaisons possibles

**Signification Sémantique**

.. code-block:: python

   # Exemple d'interprétation
   if hist[0] > 0.3:  # Bin (0,0,0) - couleurs sombres
       print("Objet contient beaucoup de noir/sombre")
   if hist[47] > 0.2:  # Bin (3,3,2) - couleurs claires
       print("Objet contient du blanc/clair")

2. Texture LBP (10 dimensions)
------------------------------

**Principe**
    Local Binary Pattern analyse les micro-textures en comparant chaque pixel avec ses 8 voisins, créant des patterns binaires locaux.

**Algorithme LBP**

.. code-block:: python

   def extract_lbp_features(roi):
       # Conversion en niveaux de gris
       gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
       
       # Calcul LBP uniforme
       lbp = local_binary_pattern(
           gray, 
           P=8,              # 8 voisins
           R=1,              # Rayon 1
           method='uniform'  # Patterns uniformes seulement
       )
       
       # Histogramme des patterns uniformes (0-9)
       hist, _ = np.histogram(
           lbp.ravel(), 
           bins=np.arange(0, 11),  # 11 bins pour 0-10
           range=(0, 10)
       )
       
       # Normalisation
       hist = hist / (hist.sum() + 1e-5)  # Évite division par 0
       return hist  # Shape: (10,)

**Interprétation des Patterns**
    * **Pattern 0** : Centre plus sombre que tous les voisins
    * **Pattern 1-8** : Variations de transitions sombre→clair
    * **Pattern 9** : Patterns uniformes complexes
    * **Pattern 10** : Autres patterns non-uniformes

**Applications**
    * Détection de textures lisses vs rugueuses
    * Identification de patterns réguliers
    * Caractérisation de surfaces (métal, bois, tissu, etc.)

3. Caractéristiques Géométriques (3 dimensions)
-----------------------------------------------

**3.1 Aire Relative (1 dimension)**

.. code-block:: python

   def calculate_relative_area(box, image_size):
       x1, y1, x2, y2 = box
       width_img, height_img = image_size
       
       # Aire de la bounding box
       box_area = (x2 - x1) * (y2 - y1)
       
       # Aire totale de l'image
       total_area = width_img * height_img
       
       # Pourcentage de l'image occupé
       relative_area = box_area / total_area
       return relative_area  # Valeur entre 0 et 1

**3.2 Aspect Ratio (1 dimension)**

.. code-block:: python

   def calculate_aspect_ratio(box):
       x1, y1, x2, y2 = box
       
       width = x2 - x1
       height = y2 - y1
       
       # Ratio hauteur/largeur
       aspect_ratio = height / (width + 1e-5)  # Évite division par 0
       return aspect_ratio

**Interprétation**
    * **aspect_ratio < 1** : Objet plus large que haut (horizontal)
    * **aspect_ratio = 1** : Objet carré
    * **aspect_ratio > 1** : Objet plus haut que large (vertical)

**3.3 Compacité/Circularité (1 dimension)**

.. code-block:: python

   def calculate_compactness(mask):
       # Calcul du périmètre et de l'aire du masque
       contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
       if len(contours) > 0:
           largest_contour = max(contours, key=cv2.contourArea)
           area = cv2.contourArea(largest_contour)
           perimeter = cv2.arcLength(largest_contour, True)
           
           # Compacité : 4π × aire / périmètre²
           # Cercle parfait = 1, formes allongées < 1
           compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-5)
           return compactness
       return 0.0

Intégration et Usage
--------------------

**Pipeline Complet d'Extraction**

.. code-block:: python

   def extract_hlbb_features(image, box):
       # Extraction ROI
       x1, y1, x2, y2 = map(int, box)
       roi = image[y1:y2, x1:x2]
       
       # 1. Histogramme couleur (48 dims)
       color_hist = extract_color_histogram(roi)
       
       # 2. Texture LBP (10 dims) 
       lbp_hist = extract_lbp_features(roi)
       
       # 3. Géométrie (3 dims)
       rel_area = calculate_relative_area(box, image.shape[:2])
       aspect_ratio = calculate_aspect_ratio(box)
       compactness = calculate_compactness_from_box(roi)
       
       # Concaténation finale
       features = np.concatenate([
           color_hist,      # 48 dims
           lbp_hist,        # 10 dims
           [rel_area, aspect_ratio, compactness]  # 3 dims
       ])
       
       return features  # Shape: (61,)

**Format de Sortie JSON**

.. code-block:: python

   hlbb_output = {
       "box": [x1, y1, x2, y2],
       "features": {
           "color_histogram": color_hist.tolist(),  # 48 valeurs
           "texture_lbp": lbp_hist.tolist(),       # 10 valeurs
           "relative_area": float(rel_area),        # 1 valeur
           "aspect_ratio": float(aspect_ratio),     # 1 valeur
           "compactness": float(compactness)        # 1 valeur
       },
       "metadata": {
           "total_dimensions": 61,
           "extraction_time": timestamp
       }
   }

Applications des Features HLBB
------------------------------

**1. Classification d'Objets**

.. code-block:: python

   # Utilisation en Machine Learning
   from sklearn.ensemble import RandomForestClassifier
   
   # Entraînement d'un classifieur
   clf = RandomForestClassifier()
   clf.fit(hlbb_features_matrix, object_labels)
   
   # Prédiction sur nouveaux objets
   predicted_class = clf.predict(new_hlbb_features)

**2. Recherche par Similarité**

.. code-block:: python

   # Distance euclidienne entre features
   def calculate_similarity(features1, features2):
       return np.linalg.norm(features1 - features2)
   
   # Recherche d'objets similaires
   similarities = []
   for obj_features in database:
       sim = calculate_similarity(query_features, obj_features)
       similarities.append(sim)

**3. Clustering d'Objets**

.. code-block:: python

   from sklearn.cluster import KMeans
   
   # Regroupement automatique d'objets similaires
   kmeans = KMeans(n_clusters=5)
   clusters = kmeans.fit_predict(hlbb_features_matrix)

**4. Analyse Statistique**

.. code-block:: python

   # Analyse des distributions de features
   import matplotlib.pyplot as plt
   
   # Distribution des aires relatives
   plt.hist([feat[58] for feat in hlbb_features], bins=20)
   plt.title("Distribution des Aires Relatives")
   
   # Corrélation entre aspect ratio et compacité
   plt.scatter([feat[59] for feat in hlbb_features], 
              [feat[60] for feat in hlbb_features])
   plt.xlabel("Aspect Ratio")
   plt.ylabel("Compacité")

Performance et Optimisation
---------------------------

**Complexité Computationnelle**
    * **Histogramme Couleur** : O(W×H) - Linéaire en nombre de pixels
    * **LBP** : O(W×H×8) - 8 comparaisons par pixel
    * **Géométrie** : O(1) - Calculs constants

**Optimisations**

.. code-block:: python

   # Redimensionnement ROI pour accélérer
   def resize_roi_if_large(roi, max_size=256):
       h, w = roi.shape[:2]
       if max(h, w) > max_size:
           scale = max_size / max(h, w)
           new_h, new_w = int(h * scale), int(w * scale)
           roi = cv2.resize(roi, (new_w, new_h))
       return roi

**Parallélisation**

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   
   # Traitement parallèle de multiples ROIs
   def extract_features_parallel(image, boxes):
       with ThreadPoolExecutor(max_workers=4) as executor:
           futures = [
               executor.submit(extract_hlbb_features, image, box) 
               for box in boxes
           ]
           return [future.result() for future in futures]

Validation et Qualité
---------------------

**Tests de Robustesse**

.. code-block:: python

   # Test d'invariance aux transformations
   def test_feature_robustness(image, box):
       original_features = extract_hlbb_features(image, box)
       
       # Rotation légère
       rotated_image = rotate_image(image, angle=5)
       rotated_features = extract_hlbb_features(rotated_image, box)
       
       # Changement d'éclairage
       bright_image = adjust_brightness(image, factor=1.2)
       bright_features = extract_hlbb_features(bright_image, box)
       
       # Calcul de stabilité
       rotation_stability = cosine_similarity(original_features, rotated_features)
       brightness_stability = cosine_similarity(original_features, bright_features)
       
       return rotation_stability, brightness_stability

**Métriques de Qualité**

.. code-block:: python

   def evaluate_feature_quality(features_matrix, labels):
       # Discriminativité inter-classes
       inter_class_distance = calculate_inter_class_distance(features_matrix, labels)
       
       # Cohérence intra-classe
       intra_class_variance = calculate_intra_class_variance(features_matrix, labels)
       
       # Ratio de séparabilité
       separability_ratio = inter_class_distance / intra_class_variance
       
       return separability_ratio