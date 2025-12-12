"""設定管理モジュール"""
import os
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

# 企業名（現在は固定、将来は選択式に）
COMPANY_NAME = "Starbucks Coffee Japan"

# 公式コメントを判定するguest_id（数値型・文字列型の両方に対応）
OFFICIAL_GUEST_ID = "555674619"

# Anthropic API設定
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# OpenAI API設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Sheets API設定
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

# チャットの属性リスト
CHAT_ATTRIBUTES = [
    "公式コメント",
    "商品に対する質問",
    "出演者に対する質問",
    "商品に対するリアクション",
    "出演者に対するリアクション",
    "商品と出演者に対するリアクション",
    "配信に対するリアクション",
    "絵文字のみ",
    "不満の声",
    "購入検討",
    "購入報告",
    "お礼・感謝",
    "その他"
]

# チャット感情リスト
CHAT_SENTIMENTS = [
    "ポジティブ",
    "ややポジティブ",
    "どちらでもない",
    "ややネガティブ",
    "ネガティブ",
    "混在"
]

# 回答状況リスト
ANSWER_STATUSES = [
    "出演者",
    "運営",
    "未回答"
]

# 色分け設定（Google Sheets用）
COLOR_MAP = {
    # チャットの属性の色
    "公式コメント": {"red": 0.8, "green": 0.9, "blue": 1.0},
    "商品に対する質問": {"red": 1.0, "green": 0.95, "blue": 0.8},
    "出演者に対する質問": {"red": 0.95, "green": 0.85, "blue": 1.0},
    "商品に対するリアクション": {"red": 0.9, "green": 1.0, "blue": 0.85},
    "出演者に対するリアクション": {"red": 0.85, "green": 0.9, "blue": 1.0},
    "商品と出演者に対するリアクション": {"red": 1.0, "green": 0.9, "blue": 0.85},
    "配信に対するリアクション": {"red": 0.95, "green": 0.95, "blue": 1.0},
    "絵文字のみ": {"red": 0.95, "green": 0.95, "blue": 0.95},
    "不満の声": {"red": 1.0, "green": 0.7, "blue": 0.7},
    "購入検討": {"red": 1.0, "green": 0.95, "blue": 0.8},
    "購入報告": {"red": 0.85, "green": 1.0, "blue": 0.85},
    "お礼・感謝": {"red": 0.85, "green": 0.95, "blue": 0.9},
    "その他": {"red": 0.9, "green": 0.9, "blue": 0.9},
    # チャット感情の色
    "ポジティブ": {"red": 0.85, "green": 1.0, "blue": 0.85},
    "ややポジティブ": {"red": 0.9, "green": 0.98, "blue": 0.9},
    "どちらでもない": {"red": 0.9, "green": 0.9, "blue": 0.9},
    "ややネガティブ": {"red": 0.98, "green": 0.9, "blue": 0.9},
    "ネガティブ": {"red": 1.0, "green": 0.85, "blue": 0.85},
    "混在": {"red": 1.0, "green": 0.95, "blue": 0.7},
    # 回答状況の色
    "出演者": {"red": 0.7, "green": 1.0, "blue": 0.7},
    "運営": {"red": 0.7, "green": 0.85, "blue": 1.0},
    "未回答": {"red": 0.95, "green": 0.95, "blue": 0.95}
}

# 企業設定
COMPANIES = {
    "Starbucks Coffee Japan": {
        "name": "Starbucks Coffee Japan",
        "chat_attributes": CHAT_ATTRIBUTES,
        "chat_sentiments": CHAT_SENTIMENTS,
        "official_user_type": "moderator",  # user_typeが"moderator"の場合に公式コメント判定
        "official_guest_id": None,  # guest_idによる判定は使用しない
        "official_username": None  # usernameによる判定は使用しない
    },
    "マツココライブ": {
        "name": "マツココライブ",
        "chat_attributes": CHAT_ATTRIBUTES,
        "chat_sentiments": CHAT_SENTIMENTS,
        "official_user_type": "moderator",
        "official_guest_id": None,
        "official_username": ["マツキヨココカラSTAFF"]
    },
    "ヤマダライブ": {
        "name": "ヤマダライブ",
        "chat_attributes": CHAT_ATTRIBUTES,
        "chat_sentiments": CHAT_SENTIMENTS,
        "official_user_type": "moderator",
        "official_guest_id": None,
        "official_username": None
    }
}

# デフォルトの企業名
DEFAULT_COMPANY = "Starbucks Coffee Japan"


def get_company_config(company_name: str) -> dict:
    """
    企業設定を取得
    
    Args:
        company_name: 企業名
        
    Returns:
        企業設定の辞書
    """
    return COMPANIES.get(company_name, COMPANIES[DEFAULT_COMPANY])


def get_current_company_config():
    """
    現在選択されている企業の設定を取得（セッションステートから）

    Returns:
        企業設定の辞書
    """
    import streamlit as st
    selected_company = st.session_state.get("selected_company", DEFAULT_COMPANY)
    return get_company_config(selected_company)


def get_openai_api_key():
    """
    有効なOpenAI APIキーを取得

    優先順位:
    1. セッションステート（ユーザー入力）
    2. ローカルストレージ（記憶されたキー）
    3. 環境変数

    Returns:
        有効なAPIキー、なければNone
    """
    try:
        from utils.api_key_manager import get_active_api_key
        return get_active_api_key()
    except ImportError:
        # フォールバック: 環境変数のみ
        return os.getenv("OPENAI_API_KEY")
