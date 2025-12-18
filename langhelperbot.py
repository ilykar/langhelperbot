import telebot
from googletrans import Translator
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# импорт модуля достопримечательностей
from landmarks import find_landmark_info

# настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# загрузка конфигурации
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

# проверка токена
if not TOKEN:
    print("ОШИБКА: токен не найден")
    exit(1)

if ":" not in TOKEN:
    print("ОШИБКА: неверный формат токена")
    exit(1)

# инициализация бота и переводчика
bot = telebot.TeleBot(TOKEN)
translator = Translator()

# инициализация нейросети OCR
print("Инициализация нейросети EasyOCR...")
try:
    reader_europe = easyocr.Reader(['en', 'ru'], gpu=False)
    reader_japanese = easyocr.Reader(['ja', 'en'], gpu=False)
    reader_korean = easyocr.Reader(['ko', 'en'], gpu=False)
    reader_other = easyocr.Reader(['en', 'de', 'fr', 'es'], gpu=False)
    
    readers = {
        'cyrillic': reader_europe,
        'japanese': reader_japanese,
        'korean': reader_korean,
        'europe': reader_other
    }
    
    print("Нейросеть OCR загружена")
except Exception as e:
    print(f"Ошибка загрузки OCR: {e}")
    try:
        readers = {'english': easyocr.Reader(['en'], gpu=False)}
        print("Загружен только английский")
    except:
        print("Критическая ошибка: не удалось загрузить OCR")
        exit(1)

# константы
DB_FILE = "langhelper.db"

# функц бд

def init_db():
    """Инициализация базы данных"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # таб пользователей
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            target_language TEXT DEFAULT 'ru',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # таб истории
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT,
            original_text TEXT,
            translated_text TEXT,
            source_lang TEXT,
            target_lang TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print("База данных инициализирована")
    except Exception as e:
        print(f"Ошибка БД: {e}")

def add_user(user_id, username="", first_name=""):
    """Добавление пользователя"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR IGNORE INTO users (user_id, username, first_name) 
        VALUES (?, ?, ?)
        ''', (user_id, username, first_name))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка добавления пользователя: {e}")

def add_to_history(user_id, type_, original, translated, src_lang, target_lang):
    """Добавление в историю"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO history (user_id, type, original_text, translated_text, source_lang, target_lang)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, type_, original[:1000], translated[:1000], src_lang, target_lang))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка добавления в историю: {e}")

def get_user_language(user_id):
    """Получение языка пользователя"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT target_language FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 'ru'
    except:
        return 'ru'

def set_user_language(user_id, lang):
    """Установка языка"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO users (user_id, target_language) 
        VALUES (?, ?)
        ''', (user_id, lang))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка установки языка: {e}")

# инициализация бд
init_db()

# клавиатуры

def get_main_keyboard():
    """Главное меню"""
    from telebot import types
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row("📸 Распознать фото", "📝 Переводчик")
    markup.row("🌍 Язык перевода", "📚 История")
    markup.row("❓ Помощь", "🏛️ Примеры достопримечательностей")
    return markup

def get_lang_keyboard():
    """Выбор языка"""
    from telebot import types
    markup = types.InlineKeyboardMarkup(row_width=3)
    languages = [
        ("🇬🇧 Английский", "en"),
        ("🇩🇪 Немецкий", "de"),
        ("🇫🇷 Французский", "fr"),
        ("🇪🇸 Испанский", "es"),
        ("🇯🇵 Японский", "ja"),
        ("🇰🇷 Корейский", "ko"),
        ("🇷🇺 Русский", "ru"),
        ("🇮🇹 Итальянский", "it"),
        ("🇵🇹 Португальский", "pt"),
        ("🇦🇪 Арабский", "ar"),
        ("🇹🇷 Турецкий", "tr"),
        ("🇨🇳 Китайский", "zh-cn")
    ]
    for name, code in languages:
        markup.add(types.InlineKeyboardButton(name, callback_data=f"lang_{code}"))
    return markup

# функц оцр

def process_image_ocr(image_bytes):
    """Обработка изображения и распознавание текста"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(image)
        
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        all_results = []
        
        # пробуем сначала кириллический читатель (для русского)
        try:
            if 'cyrillic' in readers:
                result = readers['cyrillic'].readtext(img_np, detail=0, paragraph=True)
                if result:
                    text = ' '.join(result).strip()
                    if text and len(text) > 1:
                        all_results.append(('cyrillic', text))
        except Exception as e:
            logger.error(f"Ошибка кириллического OCR: {e}")
        
        # пробуем другие читатели
        for reader_name, reader in readers.items():
            if reader_name == 'cyrillic':
                continue
                
            try:
                result = reader.readtext(img_np, detail=0, paragraph=True)
                if result:
                    text = ' '.join(result).strip()
                    if text and len(text) > 1:
                        all_results.append((reader_name, text))
            except Exception as e:
                logger.error(f"Ошибка OCR {reader_name}: {e}")
        
        if not all_results:
            return None
        
        # выбираем самый длинный результат
        best_result = max(all_results, key=lambda x: len(x[1]))
        return best_result[1]
        
    except Exception as e:
        logger.error(f"Общая ошибка OCR: {e}")
        return None

# обтработчики команд

@bot.message_handler(commands=['start'])
def cmd_start(message):
    """Команда старт"""
    user_id = message.from_user.id
    username = message.from_user.username or ""
    first_name = message.from_user.first_name or ""
    
    add_user(user_id, username, first_name)
    set_user_language(user_id, 'ru')
    
    welcome = f"""
Привет, {first_name}! 🎉 Я ИИ-переводчик для путешествий

Я помогу тебе:
📸 Распознаю текст с фото (вывески, меню, указатели)
🏛️ Определю достопримечательности по названию
🌍 Переведу на 100+ языков
📚 Сохраню историю переводов

**Как пользоваться:**
1. Отправь фото с текстом → получи перевод
2. Напиши название достопримечательности → узнай о ней
3. Напиши любой текст → получи перевод

Примеры достопримечательностей:
Эйфелева башня, Красная площадь, Колизей, Статуя Свободы, Тадж-Махал
    """
    
    bot.send_message(message.chat.id, welcome, 
                    reply_markup=get_main_keyboard(),
                    parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def cmd_help(message):
    """Команда помощь"""
    help_text = """
📚 **Помощь по использованию:**

📸 **Для фото с текстом:**
1. Сфотографируйте текст (вывеску, меню, указатель)
2. Отправьте фото боту
3. Получите перевод или информацию о достопримечательности

🏛️ **Для определения достопримечательностей:**
• Напишите название (например: "Эйфелева башня")
• Или отправьте фото с названием достопримечательности

🌍 **Для перевода текста:**
• Просто напишите текст на любом языке
• Бот определит язык и переведёт

📋 **Примеры запросов:**
• Фото с надписью "Eiffel Tower"
• Текст "Красная площадь Москва"
• "Where is Colosseum?" (переведёт и найдёт Колизей)

🔧 **Команды:**
/start - начало работы
/help - эта справка  
/language - выбрать язык перевода
/history - показать историю
/clear - очистить историю
/examples - примеры достопримечательностей
    """
    
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['examples'])
def cmd_examples(message):
    """Примеры достопримечательностей"""
    examples = """
🏛️ **Примеры достопримечательностей для поиска:**

**Россия:**
• Красная площадь
• Московский Кремль
• Эрмитаж
• Петергоф
• Собор Василия Блаженного

**Европа:**
• Эйфелева башня (Eiffel Tower)
• Лувр (Louvre)
• Колизей (Colosseum)
• Биг-Бен (Big Ben)
• Собор Святого Петра

**Америка:**
• Статуя Свободы (Statue of Liberty)
• Белый дом (White House)
• Гора Рашмор

**Азия:**
• Великая Китайская стена (Great Wall of China)
• Тадж-Махал (Taj Mahal)
• Фудзияма (Mount Fuji)
• Ангкор-Ват (Angkor Wat)

**Отправьте название на русском или английском!**
    """
    
    bot.send_message(message.chat.id, examples, parse_mode='Markdown')

@bot.message_handler(commands=['language', 'lang'])
def cmd_language(message):
    """Выбор языка"""
    bot.send_message(message.chat.id, 
                    "Выберите язык для перевода:",
                    reply_markup=get_lang_keyboard(),
                    parse_mode='Markdown')

@bot.message_handler(commands=['history'])
def cmd_history(message):
    """История переводов"""
    user_id = message.from_user.id
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT type, original_text, translated_text, source_lang, target_lang, timestamp
    FROM history 
    WHERE user_id = ? 
    ORDER BY timestamp DESC 
    LIMIT 10
    ''', (user_id,))
    
    history = cursor.fetchall()
    conn.close()
    
    if not history:
        bot.send_message(message.chat.id, 
                        "История пуста",
                        parse_mode='Markdown')
        return
    
    response = "📚 **Последние 10 запросов:**\n\n"
    
    for i, (type_, orig, trans, src, targ, time) in enumerate(history, 1):
        icon = "📸" if 'photo' in type_ else "📝"
        if 'landmark' in type_:
            icon = "🏛️"
        
        orig_display = orig[:40] + "..." if len(orig) > 40 else orig
        trans_display = trans[:40] + "..." if len(trans) > 40 else trans
        
        try:
            time_str = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").strftime("%d.%m %H:%M")
        except:
            time_str = time[:16]
        
        response += f"{icon} **{i}.** `{orig_display}`\n"
        response += f"   → `{trans_display}`\n"
        response += f"   🌐 `{src.upper()} → {targ.upper()}` | 🕒 {time_str}\n\n"
    
    bot.send_message(message.chat.id, response, parse_mode='Markdown')

@bot.message_handler(commands=['clear'])
def cmd_clear(message):
    """Очистка истории"""
    user_id = message.from_user.id
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
    bot.send_message(message.chat.id, "✅ История очищена")

# обработка фото

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """Обработка фото: распознаём текст и ищем достопримечательности"""
    user_id = message.from_user.id
    
    processing_msg = bot.send_message(message.chat.id, 
                                     "📸 Распознаю текст на фото...",
                                     parse_mode='Markdown')
    
    try:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # распознаём текст на фото
        recognized_text = process_image_ocr(downloaded_file)
        
        # если текст распознан
        if recognized_text and len(recognized_text.strip()) > 2:
            display_text = recognized_text[:300] + "..." if len(recognized_text) > 300 else recognized_text
            
            # пробуем найти достопримечательность в тексте
            landmark_info = find_landmark_info(recognized_text)
            
            if landmark_info['found']:
                # нашли достопримечательность
                response = f"""
🏛️ **Найдена достопримечательность!**

**{landmark_info['name']}**
{landmark_info['description']}

{landmark_info['fact']}

📝 **Текст на фото:**
`{display_text}`

🔍 Определено по распознанному тексту
                """
                
                # добавляем в историю
                add_to_history(user_id, 'photo_landmark', recognized_text[:100], 
                              landmark_info['name'], 'text', 'landmark')
                
                bot.edit_message_text(response,
                                     message.chat.id,
                                     processing_msg.message_id,
                                     parse_mode='Markdown')
                return
            
            # если достопримечательность не найдена, переводим текст
            bot.edit_message_text("🌍 Определяю язык для перевода...",
                                 message.chat.id,
                                 processing_msg.message_id,
                                 parse_mode='Markdown')
            
            detected = translator.detect(recognized_text)
            src_lang = detected.lang
            confidence = detected.confidence * 100
            
            target_lang = get_user_language(user_id)
            translation = translator.translate(recognized_text, src=src_lang, dest=target_lang)
            
            # добавляем в историю
            add_to_history(user_id, 'photo', recognized_text, translation.text, src_lang, target_lang)
            
            # формируем ответ
            lang_names = {
                'en': 'английский', 'ru': 'русский', 'de': 'немецкий',
                'fr': 'французский', 'es': 'испанский', 'zh-cn': 'китайский',
                'ja': 'японский', 'ko': 'корейский', 'it': 'итальянский',
                'pt': 'португальский', 'ar': 'арабский', 'tr': 'турецкий'
            }
            
            src_name = lang_names.get(src_lang, src_lang)
            targ_name = lang_names.get(target_lang, target_lang)
            
            response = f"""
📸 **Распознанный текст:**
`{display_text}`

🌐 **Язык:** {src_name.upper()} (точность: {confidence:.1f}%)
➡️ **Перевод на {targ_name.upper()}:**
{translation.text}
            """
            
            bot.edit_message_text(response,
                                 message.chat.id,
                                 processing_msg.message_id,
                                 parse_mode='Markdown')
            
        else:
            # не удалось распознать текст
            bot.edit_message_text("❌ Не удалось распознать текст на фото.\n\n"
                                "**Советы для лучшего распознавания:**\n"
                                "• Убедитесь, что текст хорошо освещён\n"
                                "• Текст должен быть чётким и контрастным\n"
                                "• Попробуйте сфотографировать под прямым углом\n"
                                "• Избегайте бликов и отражений\n\n"
                                "Можете также просто написать текст для перевода.",
                                 message.chat.id,
                                 processing_msg.message_id,
                                 parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}")
        error_msg = f"❌ Ошибка обработки фото: `{str(e)[:100]}`"
        bot.edit_message_text(error_msg,
                             message.chat.id,
                             processing_msg.message_id,
                             parse_mode='Markdown')

# обработ текста

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    """Обработка текста: ищем достопримечательности или переводим"""
    text = message.text.strip()
    user_id = message.from_user.id
    
    # проверяем команды меню
    if text == "📸 Распознать фото":
        bot.send_message(message.chat.id,
                        "Отправьте фото с текстом для распознавания и перевода.\n\n"
                        "📌 **Совет:** Фотографируйте текст чётко, без бликов.",
                        parse_mode='Markdown')
        return
        
    elif text == "📝 Переводчик":
        bot.send_message(message.chat.id,
                        "Напишите текст для перевода на выбранный язык.\n\n"
                        "Пример: 'Hello, how are you?' или 'Привет, как дела?'",
                        parse_mode='Markdown')
        return
        
    elif text == "🌍 Язык перевода":
        cmd_language(message)
        return
        
    elif text == "📚 История":
        cmd_history(message)
        return
        
    elif text == "❓ Помощь":
        cmd_help(message)
        return
        
    elif text == "🏛️ Примеры достопримечательностей":
        cmd_examples(message)
        return
    
    if len(text) < 2:
        bot.send_message(message.chat.id,
                        "Текст слишком короткий. Напишите минимум 2 символа.",
                        parse_mode='Markdown')
        return
    
    # пробуем найти достопримечательность в тексте
    landmark_info = find_landmark_info(text)
    
    if landmark_info['found']:
        # это достопримечательность
        response = f"""
🏛️ **Достопримечательность найдена!**

**{landmark_info['name']}**
{landmark_info['description']}

{landmark_info['fact']}

📌 Английское название: {landmark_info.get('en_name', 'Нет информации')}

💡 *Можете также отправить фото с названием этой достопримечательности*
        """
        
        # добавляем в историю
        add_to_history(user_id, 'text_landmark', text, landmark_info['name'], 'landmark', 'info')
        
        bot.reply_to(message, response, parse_mode='Markdown')
        return
    
    # если не достопримечательность, делаем перевод
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        
        detected = translator.detect(text)
        src_lang = detected.lang
        confidence = detected.confidence * 100
        
        target_lang = get_user_language(user_id)
        translation = translator.translate(text, src=src_lang, dest=target_lang)
        
        add_to_history(user_id, 'text', text, translation.text, src_lang, target_lang)
        
        lang_names = {
            'en': 'английский', 'ru': 'русский', 'de': 'немецкий',
            'fr': 'французский', 'es': 'испанский', 'ja': 'японский',
            'ko': 'корейский', 'zh-cn': 'китайский', 'it': 'итальянский',
            'pt': 'португальский', 'ar': 'арабский', 'tr': 'турецкий'
        }
        
        src_name = lang_names.get(src_lang, src_lang)
        targ_name = lang_names.get(target_lang, target_lang)
        
        response = f"""
📝 **Исходный текст ({src_name.upper()}):**
`{text}`

🌐 **Язык:** {src_name.upper()} (точность: {confidence:.1f}%)
➡️ **Перевод на {targ_name.upper()}:**
{translation.text}

💡 *Хотите узнать о достопримечательности? Напишите её название!*
        """
        
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка перевода: `{str(e)[:100]}`", parse_mode='Markdown')

# обработчик callback

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    """Обработка callback (выбор языка)"""
    try:
        if call.data.startswith("lang_"):
            lang = call.data[5:]
            user_id = call.from_user.id
            set_user_language(user_id, lang)
            
            lang_names = {
                'en': 'английский', 'de': 'немецкий', 'fr': 'французский',
                'es': 'испанский', 'ja': 'японский', 'ko': 'корейский',
                'ru': 'русский', 'it': 'итальянский', 'pt': 'португальский',
                'ar': 'арабский', 'tr': 'турецкий', 'zh-cn': 'китайский'
            }
            
            lang_name = lang_names.get(lang, lang)
            
            bot.answer_callback_query(call.id, f"Язык перевода: {lang_name}")
            bot.edit_message_text(
                f"✅ Язык перевода установлен: **{lang_name.upper()}**\n\n"
                f"Теперь весь текст будет переводиться на {lang_name}.",
                call.message.chat.id,
                call.message.message_id,
                parse_mode='Markdown'
            )
            
    except Exception as e:
        bot.answer_callback_query(call.id, f"Ошибка: {str(e)[:50]}")

# заупск бота

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 ЗАПУСК ИИ-ПЕРЕВОДЧИКА ДЛЯ ПУТЕШЕСТВИЙ")
    print("=" * 60)
    print("📁 База данных:", DB_FILE)
    print("🔍 Поиск достопримечательностей: ✅ Включён")
    print("📸 Распознавание фото: ✅ Включено (OCR)")
    print("🌍 Поддерживаемых языков: 100+")
    print("=" * 60)
    print("\n🤖 Бот запущен! Ожидаю запросы...")
    
    try:
        bot.infinity_polling()
    except KeyboardInterrupt:
        print("\n✅ Бот остановлен пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")