import os
import logging
import sys
import telebot
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, ConfigDict
from threading import Lock
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    StaticIndexChunkingStrategy,
    HybridSearchIndexType,
    ReciprocalRankFusionIndexCombinationStrategy,
)

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
def load_environment():
    """Load environment variables with better error handling and feedback"""
    # Try multiple locations for .env file
    possible_paths = [
        '.env',  # Current directory
        os.path.abspath(os.path.join(os.path.dirname(__file__), '.env')),  # Script directory
        os.path.abspath(os.path.join(os.getcwd(), '.env')),  # Working directory
    ]
    
    # Try each path
    env_loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found .env file at: {path}")
            load_dotenv(dotenv_path=path, override=True)
            env_loaded = True
            break
    
    if not env_loaded:
        print("No .env file found in any of the expected locations.")
        print(f"Searched paths: {possible_paths}")
        print("Looking for environment variables directly...")
    
    # Check required variables
    missing_vars = []
    for var in ["folder_id", "api_key", "tg_token"]:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # Mask the value for security but show that it exists
            masked = value[:3] + '*' * (len(value) - 6) + value[-3:] if len(value) > 6 else '***'
            print(f"Found {var}: {masked}")
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file or set these environment variables directly.")
        return False
    
    print("All required environment variables loaded successfully.")
    return True

# Call this function before creating the Config object
if not load_environment():
    sys.exit(1)

class Config(BaseSettings):
    model_config = ConfigDict(extra='ignore')
    
    # Provide default values or handle missing values
    folder_id: str = Field(default=None, env="folder_id")
    api_key: str = Field(default=None, env="api_key")
    tg_token: str = Field(default=None, env="tg_token")
    
    def validate_config(self):
        """Validate all required fields are present"""
        missing = []
        for field in ["folder_id", "api_key", "tg_token"]:
            if not getattr(self, field):
                missing.append(field)
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        return True

# Then try to create the config
try:
    config = Config()
    # Then validate after creation
    config.validate_config()
except ValueError as e:
    raise ValueError(f"Error loading configuration: {e}") from e

# --- Data Processing ---
class WineDataProcessor:
    """Обрабатывает данные о винах из Excel-файла."""
    def __init__(self, data_path="data/wine-price.xlsx"):
        self.data_path = data_path
        self.wine_df = self.load_wine_data()

    def load_wine_data(self) -> pd.DataFrame:
        """Загружает данные о винах из Excel-файла."""
        try:
            logger.info(f"Loading wine data from {self.data_path}")
            df = pd.read_excel(self.data_path)
            logger.info(f"Wine data loaded successfully. Processing...")
            return self.process_wine_data(df)
        except FileNotFoundError:
            logger.error(f"Wine data file not found: {self.data_path}")
            raise ValueError(f"Wine data file not found: {self.data_path}")
        except pd.errors.ParserError:
            logger.error(f"Error parsing wine data file: {self.data_path}")
            raise ValueError(f"Error parsing wine data file: {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading wine data: {e}")
            raise ValueError(f"Failed to load wine data from {self.data_path}") from e

    def process_wine_data(self, df) -> pd.DataFrame:
        """Обрабатывает данные о винах, добавляет необходимые столбцы и очищает данные."""
        df.columns = ["Id", "Name", "Country", "Price", "WHPrice", "etc"]                                                                                                           
        acid_map = {"СХ": "Сухое", "СЛ": "Сладкое", "ПСХ": "Полусухое", "ПСЛ": "Полусладкое"}
        df["Acidity"] = df["Name"].apply(lambda x: acid_map.get(x.split()[-1].replace("КР", ""), ""))
        df["Color"] = df["Name"].apply(lambda x: "Красное" if (x.split()[-1].startswith("КР") or x.split()[-2] == "КР") else "")
        logger.info(f"Wine data processed successfully. Found {len(df)} wines")
        return df

# --- Wine Search ---
class WineSearcher:
    def __init__(self, wine_df):
        self.wine_df = wine_df
        self.country_map = {
            "IT": "Италия", "FR": "Франция", "ES": "Испания", "RU": "Россия", "PT": "Португалия", "AR": "Армения",
            "CL": "Чили", "GE": "Грузия", "ZA": "ЮАР", "US": "США", "NZ": "Новая Зеландия",
            "DE": "Германия", "AT": "Австрия", "IL": "Израиль", "BG": "Болгария", "GR": "Греция", "AU": "Австралия",
        }
        self.revmap = {v.lower(): k for k, v in self.country_map.items()}

    def filter_by_country(self, df, country):
        country_query = country.lower().strip()
        if country_query in self.revmap:
            logger.info(f"Searching for wines from {country}")
            filtered_df = df[df["Country"] == self.revmap[country_query]]
            logger.info(f"Found {len(filtered_df)} wines from {country}")
            return filtered_df
        logger.warning(f"Country '{country}' not found in the mapping.")
        return df

    def filter_by_acidity(self, df, acidity):
        logger.info(f"Searching for {acidity} wines")
        filtered_df = df[df["Acidity"] == acidity.capitalize()]
        logger.info(f"Found {len(filtered_df)} {acidity} wines")
        return filtered_df

    def filter_by_color(self, df, color):
        logger.info(f"Searching for {color} wines")
        filtered_df = df[df["Color"] == color.capitalize()]
        logger.info(f"Found {len(filtered_df)} {color} wines")
        return filtered_df

    def filter_by_name(self, df, name):
        logger.info(f"Searching for wines containing '{name}'")
        filtered_df = df[df["Name"].apply(lambda x: name.lower() in x.lower())]
        logger.info(f"Found {len(filtered_df)} wines containing '{name}'")
        return filtered_df

    def sort_wines(self, df, sort_order):
        logger.info(f"Sorting wines by {sort_order}")
        if sort_order == "cheapest":
            return df.sort_values(by="Price")
        elif sort_order == "most expensive":
            return df.sort_values(by="Price", ascending=False)
        logger.warning(f"Invalid sort order: {sort_order}. No sorting applied.")
        return df

    def format_results(self, df):
        if df.empty:
            logger.info("No wines found matching the criteria.")
            return "Подходящих вин не найдено"
        logger.info(f"Found {len(df)} wines matching the criteria.")
        return "Найдено всего наименований: " + str(len(df)) + ". Вот какие вина были найдены:\n" + "\n".join(
            [f"{z['Name']} ({self.country_map.get(z['Country'],'Неизвестно')}) - {z['Price']}" for _, z in df.head(10).iterrows()]
        )

    def find_wines(self, req):
        logger.info(f"Searching for wines matching the criteria: {req}")
        x = self.wine_df.copy()
        logger.info(f"Found {len(x)} wines in total")
        if req.country:
            x = self.filter_by_country(x, req.country)
            logger.info(f"Found {len(x)} wines after filtering by country")
        if req.acidity:
            x = self.filter_by_acidity(x, req.acidity)
            logger.info(f"Found {len(x)} wines after filtering by acidity")
        if req.color:
            x = self.filter_by_color(x, req.color)
            logger.info(f"Found {len(x)} wines after filtering by color")
        if req.name:
            x = self.filter_by_name(x, req.name)
            logger.info(f"Found {len(x)} wines after filtering by name")
        if req.sort_order and len(x) > 0:
            x = self.sort_wines(x, req.sort_order)
            logger.info(f"Sorted wines by {req.sort_order}")
        logger.info(f"Found {len(x)} wines after sorting")
        return self.format_results(x)

# --- Cart ---
class CartItem(BaseModel):
    wine_name: str
    count: int
    price: float

class Cart:
    def __init__(self):
        self.items: list[CartItem] = []

    def add_item(self, item: CartItem):
        self.items.append(item)

    def remove_item(self, item_name):
        self.items = [item for item in self.items if item.wine_name != item_name]

    def clear(self):
        self.items.clear()

    def get_items(self):
        return self.items

    def is_empty(self):
        return len(self.items) == 0

    def calculate_total_price(self):
        return sum(item.price * item.count for item in self.items)

    def format_cart(self):
        if self.is_empty():
            return "Корзина пуста"
        cart_str = "В корзине находятся следующие вина:\n"
        for item in self.items:
            cart_str += f"{item.wine_name}, число бутылок: {item.count}, цена за бутылку: {item.price}\n"
        cart_str += f"Итоговая цена: {self.calculate_total_price()}"
        return cart_str

# --- Thread Management ---
class ThreadManager:
    def __init__(self, sdk):
        self.sdk = sdk
        self.threads = {}
        self.thread_locks = {}

    def get_or_create_thread(self, chat_id):
        if chat_id not in self.thread_locks:
            self.thread_locks[chat_id] = Lock()

        with self.thread_locks[chat_id]:
            if chat_id in self.threads:
                try:
                    self.threads[chat_id].write("test")
                    logger.info(f"Повторное использование потока для chat_id: {chat_id}")
                    return self.threads[chat_id]
                except Exception as e:
                    logger.info(f"Создание нового потока для chat {chat_id}: предыдущий поток недоступен ({str(e)})")
                    # Удаляем старый поток из памяти, если он заблокирован
                    if "is locked by run" in str(e):
                        del self.threads[chat_id]
            
            logger.info(f"Создание нового потока для chat_id: {chat_id}")
            self.threads[chat_id] = self.sdk.threads.create(ttl_days=1, expiration_policy="static")
            logger.info(f"Новый поток создан: {self.threads[chat_id].id}")
            return self.threads[chat_id]
    
    def delete_thread(self, chat_id):
        if chat_id in self.threads:
            logger.info(f"Deleting thread for chat_id: {chat_id}")
            self.threads[chat_id].delete()
            del self.threads[chat_id]
            logger.info(f"Thread deleted for chat {chat_id}")

# --- Search Index ---
class SearchIndexManager:
    def __init__(self, sdk):
        self.sdk = sdk
        self.index = None

    def create_index(self, df):
        logger.info("Creating search index...")
        op = self.sdk.search_indexes.create_deferred(
            df[df["Category"] == "wines"]["Uploaded"],
            index_type=HybridSearchIndexType(
                chunking_strategy=StaticIndexChunkingStrategy(max_chunk_size_tokens=1000, chunk_overlap_tokens=100),
                combination_strategy=ReciprocalRankFusionIndexCombinationStrategy(),
            ),
        )
        self.index = op.wait()
        logger.info(f"Search index created: {self.index.id}")
        return self.index

    def add_files(self, df, category):
        if self.index is None:
            raise ValueError("Search index not created")
        logger.info(f"Adding files to search index for category: {category}")
        op = self.index.add_files_deferred(df[df["Category"] == category]["Uploaded"])
        result = op.wait()
        logger.info(f"Files added to search index for category: {category}")
        return result

# --- Agent ---
class Agent:
    def __init__(self, assistant, instruction, tools, thread_manager, wine_searcher):
        self.assistant = assistant
        self.instruction = instruction
        self.tools = {tool.__name__: tool for tool in tools}
        self.thread_manager = thread_manager
        self.wine_searcher = wine_searcher

    def __call__(self, message: str, chat_id=None):
        thread = self.thread_manager.get_or_create_thread(chat_id)
        thread.write(message)
        run = self.assistant.run(thread)
        result = run.wait()

        logger.debug(f"Результат ассистента: {result}")

        if result.tool_calls:
            logger.info(f"Обнаружены вызовы инструментов: {result.tool_calls}")
            tool_results = []
            tool_outputs = []
            for call in result.tool_calls:
                fn = self.tools.get(call.function.name)
                if fn:
                    args = call.function.arguments
                    logger.info(f"Выполнение инструмента {call.function.name} с аргументами {args}")
                    instance = fn(**args)
                    if call.function.name == "SearchWinePriceList":
                        output = instance.process(thread, wine_searcher=self.wine_searcher)
                    else:
                        output = instance.process(thread)
                    tool_results.append({"name": call.function.name, "content": output})
                    tool_outputs.append(output)
            submitted_run = run.submit_tool_results(tool_results)
            if submitted_run is None:
                logger.error("submit_tool_results вернул None. Используем результат инструмента напрямую.")
                if tool_outputs:
                    combined_output = "\n".join(str(output) for output in tool_outputs if output)
                    return f"{result.text}\n{combined_output}" if result.text else combined_output
                return result.text if result.text else "К сожалению, не удалось обработать запрос."
            result = submitted_run.wait()
            logger.info(f"Результат после выполнения инструментов: {result.text}")

        return result.text if result else "Ответ не получен"

# --- Tools ---
class Handover(BaseModel):
    """Эта функция позволяет передать диалог человеку-оператору поддержки"""
    reason: str = Field(description="Причина для вызова оператора", default="не указана")

    def process(self, thread):
        return f"Я побежала вызывать оператора, ваш {thread.id=}, причина: {self.reason}"

class AddToCart(BaseModel):
    """Эта функция позволяет положить или добавить вино в корзину"""
    wine_name: str = Field(description="Точное название вина", default=None)
    count: int = Field(description="Количество бутылок", default=1)
    price: float = Field(description="Цена вина", default=None)
    carts: dict = Field(default_factory=dict)
    def process(self, thread):
        cart = self.carts.get(thread.id)
        if cart is None:
            cart = Cart()
            self.carts[thread.id] = cart
        cart.add_item(CartItem(wine_name=self.wine_name, count=self.count, price=self.price))
        logger.info(f"Wine {self.wine_name} added to cart for thread {thread.id}, count: {self.count}, price: {self.price}")
        return f"Вино {self.wine_name} добавлено в корзину, число бутылок: {self.count} по цене {self.price}"

class ShowCart(BaseModel):
    """Эта функция позволяет показать содержимое корзины"""
    def process(self, thread):
        cart = self.carts.get(thread.id)
        if cart is None or cart.is_empty():
            return "Корзина пуста"
        return cart.format_cart()

class ClearCart(BaseModel):
    """Эта функция очищает корзину"""
    def process(self, thread):
        cart = self.carts.get(thread.id)
        if cart:
            cart.clear()
        return "Корзина очищена"

class SearchWinePriceList(BaseModel):
    """Эта функция позволяет искать вина в прайс-листе по одному или нескольким параметрам."""
    name: str = Field(description="Название вина (Кьянти, Мерло и т.д.)", default=None)
    country: str = Field(description="Страна", default=None)
    acidity: str = Field(description="Кислотность (сухое, полусухое, сладкое, полусладкое)", default=None)
    color: str = Field(description="Цвет вина (красное, белое, розовое)", default=None)
    sort_order: str = Field(description="Порядок сортировки (самое дорогое, самое дешевое) | Sort order (most expensive, cheapest)", default=None)
    what_to_return: str = Field(description="Что вернуть (информация о вине или цена)", default=None)

    def process(self, thread, wine_searcher):
        """Process the search request"""
        return wine_searcher.find_wines(self)

# --- Bot ---
class Bot:
    def __init__(self, config):
        self.config = config
        self.sdk = YCloudML(folder_id=self.config.folder_id, auth=self.config.api_key)
        self.bot = telebot.TeleBot(self.config.tg_token)
        self.thread_manager = ThreadManager(self.sdk)
        self.wine_df = None
        self.index = None
        self.wine_agent = None
        self.carts = {}
        self.shared_assistant = None
        self.wine_searcher = None
    
    def initialize_rag_old(self, df):
        logger.info("Initializing RAG...")
        # ... (load files, create df)
        
        def get_token_count(filename):
            with open(filename, "r", encoding="utf8") as f:
                return len(model.tokenize(f.read()))

        def get_file_len(filename):
            with open(filename, encoding="utf-8") as f:
                return len(f.read())

        print("Loading files...")

        d = [
            {
                "File": fn,
                "Tokens": get_token_count(fn),
                "Chars": get_file_len(fn),
                "Category": os.path.basename(os.path.dirname(fn)),
            }
            for fn in glob("data\*\*.md")
        ]

        df = pd.DataFrame(d)
        print(f"Loaded {len(df)} files")
        index_manager = SearchIndexManager(self.sdk)
        self.index = index_manager.create_index(df)
        index_manager.add_files(df, "regions")
        # ... (add food-wine pairing data)
  
        # 2. Upload files and create search index
        print("Uploading files to cloud...")

        def upload_file(filename):
            return bot.sdk.files.upload(filename, ttl_days=1, expiration_policy="static")

        df["Uploaded"] = df["File"].apply(upload_file)

        print("Creating search index...")
        op = bot.sdk.search_indexes.create_deferred(
            df[df["Category"] == "wines"]["Uploaded"],
            index_type=HybridSearchIndexType(
                chunking_strategy=StaticIndexChunkingStrategy(
                    max_chunk_size_tokens=1000,
                    chunk_overlap_tokens=100
                ),
                combination_strategy=ReciprocalRankFusionIndexCombinationStrategy(),
            ),
        )
        index = op.wait()

        # 3. Add region data
        print("Adding region data...")
        op = index.add_files_deferred(df[df["Category"] == "regions"]["Uploaded"])
        xfiles = op.wait()

        # 4. Process and add food-wine pairing data
        print("Processing food-wine pairing data...")
        with open("data/food_wine_table.md", encoding="utf-8") as f:
            food_wine = f.readlines()
        header = food_wine[:2]

        chunk_size = 600 * 3
        s = header.copy()
        uploaded_foodwine = []
        for x in food_wine[2:]:
            s.append(x)
            if len("".join(s)) > chunk_size:
                id = bot.sdk.files.upload_bytes(
                    "".join(s).encode(),
                    ttl_days=5,
                    expiration_policy="static",
                    mime_type="text/markdown",
                )
                uploaded_foodwine.append(id)
                s = header.copy()

        print("Adding food-wine pairing data...")
        op = index.add_files_deferred(uploaded_foodwine)
        xfiles = op.wait()

        print(f"RAG initialization complete with {len(df)} files")
        logger.info("RAG initialization complete with %d files", len(df))

        logger.info(f"RAG initialization complete with {len(df)} files")
        return self.index

    def initialize_rag(self, files_dir="data"):
        logger.info("Initializing RAG...")
        
        # Define the tokenize function using the SDK model
        model = self.sdk.models.completions("yandexgpt", model_version="rc")
        
        def get_token_count(filename):
            with open(filename, "r", encoding="utf8") as f:
                return len(model.tokenize(f.read()))

        def get_file_len(filename):
            with open(filename, encoding="utf-8") as f:
                return len(f.read())

        logger.info("Loading files...")
        # Use os.path.join for better path handling
        files = glob(os.path.join(files_dir, "*", "*.md"))
        
        if not files:
            logger.warning(f"No files found in {files_dir}")
            return None
            
        # Build dataframe of files
        file_data = []
        for fn in files:
            try:
                file_data.append({
                    "File": fn,
                    "Tokens": get_token_count(fn),
                    "Chars": get_file_len(fn),
                    "Category": os.path.basename(os.path.dirname(fn)),
                })
            except Exception as e:
                logger.error(f"Error processing file {fn}: {e}")
        
        df = pd.DataFrame(file_data)
        logger.info(f"Loaded {len(df)} files")
        
        # Create upload function
        def upload_file(filename):
            return self.sdk.files.upload(filename, ttl_days=1, expiration_policy="static")
        
        # Upload files
        logger.info("Uploading files to cloud...")
        df["Uploaded"] = df["File"].apply(upload_file)
        
        # Create search index
        index_manager = SearchIndexManager(self.sdk)
        self.index = index_manager.create_index(df)
        
        # Add region data
        if "regions" in df["Category"].unique():
            index_manager.add_files(df, "regions")
        
        # Process food-wine pairing data
        food_wine_path = os.path.join(files_dir, "food_wine_table.md")
        if os.path.exists(food_wine_path):
            self._process_food_wine_data(food_wine_path)
        
        logger.info(f"RAG initialization complete with {len(df)} files")
        return self.index
        
    def _process_food_wine_data(self, food_wine_path):
        """Process and add food-wine pairing data as a separate method"""
        logger.info("Processing food-wine pairing data...")
        with open(food_wine_path, encoding="utf-8") as f:
            food_wine = f.readlines()
        
        header = food_wine[:2]
        chunk_size = 600 * 3
        s = header.copy()
        uploaded_foodwine = []
        
        for x in food_wine[2:]:
            s.append(x)
            if len("".join(s)) > chunk_size:
                id = self.sdk.files.upload_bytes(
                    "".join(s).encode(),
                    ttl_days=5,
                    expiration_policy="static",
                    mime_type="text/markdown",
                )
                uploaded_foodwine.append(id)
                s = header.copy()
        
        # Upload any remaining content
        if len(s) > len(header):
            id = self.sdk.files.upload_bytes(
                "".join(s).encode(),
                ttl_days=5,
                expiration_policy="static",
                mime_type="text/markdown",
            )
            uploaded_foodwine.append(id)
        
        logger.info(f"Adding {len(uploaded_foodwine)} food-wine pairing chunks...")
        if self.index and uploaded_foodwine:
            op = self.index.add_files_deferred(uploaded_foodwine)
            op.wait()

    def initialize_wine_data(self, data_path):
        logger.info("Initializing wine data...")
        wine_processor = WineDataProcessor(data_path)
        self.wine_df = wine_processor.wine_df
        logger.info(f"Loaded {len(self.wine_df)} wines")
        return self.wine_df

    def initialize_agent(self):
        logger.info("Инициализация агента...")
        '''
        instruction = """
        Ты — опытный сомелье, работающий в винном магазине. Твоя задача:
        - Отвечать на вопросы пользователя о винах,
        - Рекомендовать вина к блюдам, используя данные из поискового индекса для точных сочетаний,
        - Искать вина в прайс-листе магазина с помощью инструмента SearchWinePriceList,
        - Проактивно предлагать подходящие вина в зависимости от потребностей клиента,
        - Помогать пользователю оформить покупку (добавить в корзину с помощью AddToCart).

        Если пользователь написал команду /start:
        - Приветствуй его: "Здравствуйте! Я ваш виртуальный сомелье. Чем могу помочь?"
        - Предложи сочетание еды и вина, например: "К пасте с морепродуктами подойдет Совиньон Блан."
        - Вызови SearchWinePriceList(name="Совиньон Блан", what_to_return="информация о вине").

        Если в сообщении пользователя упоминается:
        - Блюдо или ингредиент (например, "Утка тушёная"):
        - Используй инструмент поиска в индексе для рекомендации вина, затем вызови SearchWinePriceList для проверки наличия.
        - Регион (например, "Абруццо"):
        - Используй инструмент поиска в индексе для поиска популярных вин региона, затем вызови SearchWinePriceList(name=<название вина>, what_to_return="информация о вине").
        - Название вина (например, "Мерло", "Шардоне", "Каберне"):
        - Вызови SearchWinePriceList(name=<название вина>, what_to_return="информация о вине").
        - Указание на покупку (например, "добавить в корзину Мерло"):
        - Вызови AddToCart(wine_name=<название вина>, count=1.0, price=<цена из прайс-листа>).

        Важно:
        - Для вопросов о сочетании еды и вина всегда используй инструмент поиска в индексе.
        - Всегда вызывай инструменты напрямую через их функциональность.
        - Никогда не вставляй текст вызова инструмента (например, "[SearchWinePriceList(...)]" или JSON) в ответ.
        - Если вино не найдено, сообщи: "К сожалению, <название> не найдено в прайс-листе."
        - Если что-то непонятно, задай уточняющий вопрос.
        """
        '''
        instruction = """
        Ты - опытный сомелье, продающий вино в магазине. Твоя задача - отвечать на вопросы пользователя
        про вина, рекомендовать лучшие вина к еде, а также искать вина в прайс-листе нашего магазина,
        а также проактивно предлагать пользователю приобрести вина, отвечающие его потребностям. В ответ
        на сообщение /start поинтересуйся, что нужно пользователю, предложи ему какой-то
        интересный вариант сочетания еды и вине, и попытайся продать ему вино.
        Посмотри на всю имеющуюся в твоем распоряжении информацию
        и выдай одну или несколько лучших рекомендаций.
        Если вопрос касается конкретных вин или цены, то вызови функцию SearchWinePriceList.
        Для передачи управления оператору - вызови фукцию Handover. Для добавления вина в корзину
        используй AddToCart. Для просмотра корзины: ShowCart.
        Если что-то непонятно, то лучше уточни информацию у пользователя.
        """

        self.wine_searcher = WineSearcher(self.wine_df)
        logger.info("Создан поисковик вин")

        model = self.sdk.models.completions("yandexgpt", model_version="rc")
        logger.info(f"Используется модель: {model}")

        tool_classes = [SearchWinePriceList, Handover, AddToCart, ShowCart, ClearCart]
        assistant_tools = [self.sdk.tools.function(tool) for tool in tool_classes]
        
        # Добавляем инструмент поиска по индексу
        if self.index:
            assistant_tools.append(self.sdk.tools.search_index(self.index))
            logger.info(f"Добавлен инструмент поиска по индексу: {self.index.id}")

        logger.info("Создание общего ассистента...")
        self.shared_assistant = self.sdk.assistants.create(
            model=model,
            instruction=instruction,
            tools=assistant_tools,
            ttl_days=1,
            expiration_policy="since_last_active"
        )
        logger.info(f"Создан общий ассистент: {self.shared_assistant.id}")

        self.wine_agent = Agent(
            assistant=self.shared_assistant,
            instruction=instruction,
            tools=tool_classes,
            thread_manager=self.thread_manager,
            wine_searcher=self.wine_searcher
        )
        logger.info("Агент успешно инициализирован")
        return self.wine_agent
    
    def get_assistant(self, chat_id):
        
        logger.info(f"Получение ассистента для чата {chat_id}")
        return self.wine_agent
    
    def handle_start_command(self, message):
        try:
            t = self.thread_manager.get_or_create_thread(message.chat.id)
            logger.info(f"Начинаем работу на потоке {t.id=}")
            
            assistant = self.get_assistant(message.chat.id)
            logger.info(f"Начинаем работу на потоке {t.id=}, сообщение={message.text}")

            t.write("Новый пользователь начал разговор")
            result = assistant(message="/start", chat_id=message.chat.id)
            logger.info(f"Результат ассистента: {result}")
            response = result if result else "Здравствуйте! Я - ваш персональный сомелье. Как я могу помочь вам выбрать вино?"
            self.bot.send_message(message.chat.id, response)
        except Exception as e:
            logger.error(f"Ошибка в handle_start_command: {e}")
            self.handle_error(message, e)

    def handle_stop_command(self, message):
        t = self.thread_manager.get_or_create_thread(message.chat.id)
        t.delete()
        del self.thread_manager.threads[message.chat.id]
        self.bot.send_message(message.chat.id, "Ваш разговор завершен. Если у вас есть другие вопросы, не стесняйтесь обращаться!")
        logger.info(f"Thread {t.id} deleted for chat {message.chat.id}")

    def handle_message(self, message):
        try:
            t = self.thread_manager.get_or_create_thread(message.chat.id)
            logger.info(f"Обработка сообщения на потоке {t.id=}, сообщение={message.text}")
            answer = self.wine_agent(message.text, chat_id=message.chat.id)
            logger.info(f"Ответ: {answer}")
            self.bot.send_message(message.chat.id, answer)
        except Exception as e:
            logger.error(f"Ошибка в обработчике сообщений: {e}")
            self.handle_error(message, e)

    def handle_error(self, message, e):
        logger.error(f"Error in handler: {e}")
        self.bot.send_message(
            message.chat.id,
            "Здравствуйте! К сожалению, возникли технические проблемы. Попробуйте начать разговор через минуту."
        )

    def run(self):
        self.bot.message_handler(commands=["start"])(self.handle_start_command)
        self.bot.message_handler(commands=["stop"])(self.handle_stop_command)
        self.bot.message_handler(func=lambda message: True)(self.handle_message)
        self.bot.polling(none_stop=True)
    
    def shutdown(self):
        """Properly shut down the bot and clean up resources"""
        logger.info("Shutting down bot...")
        # Stop polling if active
        self.bot.stop_polling()
        
        # Clean up threads
        threads_to_delete = list(self.thread_manager.threads.keys())
        for chat_id in threads_to_delete:
            try:
                self.thread_manager.delete_thread(chat_id)
            except Exception as e:
                logger.error(f"Error deleting thread for chat {chat_id}: {e}")
        
        logger.info("Bot shutdown complete")

# --- Main ---

def main_old():
    global carts
    carts = {}
    try:

        bot = Bot(config)
        

        bot.initialize_rag(df)
        bot.initialize_wine_data("data/wine-price.xlsx")
        bot.initialize_agent()
        bot.run()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")

# Improve main function with proper signal handling
def main():
    bot = None
    import signal
    
    def signal_handler(sig, frame):
        logger.info("Получен сигнал завершения")
        if bot:
            bot.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        bot = Bot(config)
        bot.initialize_wine_data("data/wine-price.xlsx")
        bot.initialize_rag("data")
        bot.initialize_agent()
        bot.run()
    except Exception as e:
        logger.error(f"Не удалось запустить бот: {e}")
        if bot:
            bot.shutdown()

if __name__ == "__main__":
    main()
