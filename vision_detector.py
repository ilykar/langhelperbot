import io
import os
import logging
from google.cloud import vision
from landmarks import LANDMARKS_EN, LANDMARKS_RU, LANDMARK_FACTS

logger = logging.getLogger(__name__)

# проверяем наличие ключа
KEY_FILE = "vision_key.json"

if not os.path.exists(KEY_FILE):
    print(f"Внимание: Файл ключа {KEY_FILE} не найден!")
    print("Скачай ключ из Google Cloud Console и помести в папку с ботом")
    HAS_VISION = False
else:
    HAS_VISION = True
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_FILE

def detect_landmarks(image_bytes):
    """Определяет достопримечательности на фото с помощью Google Vision API"""
    
    if not HAS_VISION:
        return None
    
    try:
        # создаём клиент Google Vision
        client = vision.ImageAnnotatorClient()
        
        # создаём объект изображения
        image = vision.Image(content=image_bytes)
        
        # выполняем обнаружение достопримечательностей
        response = client.landmark_detection(image=image)
        landmarks = response.landmark_annotations
        
        if response.error.message:
            logger.error(f"Ошибка Google Vision: {response.error.message}")
            return None
        
        if landmarks:
            results = []
            for landmark in landmarks:
                # получаем название и координаты
                name = landmark.description
                confidence = landmark.score * 100  # Процент уверенности
                
                # ищем в наших словарях
                ru_name = None
                
                # ищем в английском словаре
                name_lower = name.lower()
                for en_key, info in LANDMARKS_EN.items():
                    if en_key in name_lower or name_lower in en_key:
                        ru_name = info['ru_name']
                        break
                
                # если не нашли, пробуем найти по части названия
                if not ru_name:
                    for en_key, info in LANDMARKS_EN.items():
                        if any(word in name_lower for word in en_key.split()):
                            ru_name = info['ru_name']
                            break
                
                # формируем результат
                if ru_name:
                    ru_key = ru_name.lower()
                    fact = LANDMARK_FACTS.get(ru_key, 'Интересный факт: Эта достопримечательность имеет богатую историю')
                    
                    # получаем описание
                    description = "Описание не найдено"
                    if ru_key in LANDMARKS_RU:
                        description = LANDMARKS_RU[ru_key]['description']
                    
                    results.append({
                        'name': ru_name,
                        'english_name': name,
                        'description': description,
                        'fact': fact,
                        'confidence': confidence,
                        'locations': landmark.locations
                    })
            
            if results:
                # сортируем по уверенности и возвращаем лучший результат
                results.sort(key=lambda x: x['confidence'], reverse=True)
                return results[0]
        
        return None
        
    except Exception as e:
        logger.error(f"Ошибка в detect_landmarks: {e}")
        return None

def safe_detect_landmarks(image_bytes):
    """Безопасная версия с обработкой ошибок"""
    try:
        return detect_landmarks(image_bytes)
    except Exception as e:
        logger.error(f"Безопасная детекция не удалась: {e}")
        return None