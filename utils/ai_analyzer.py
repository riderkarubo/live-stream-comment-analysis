"""AI分析モジュール"""
import openai
from typing import Dict, Optional
import time
import json
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from config import OPENAI_API_KEY, CHAT_ATTRIBUTES, CHAT_SENTIMENTS, COMPANY_NAME, OFFICIAL_GUEST_ID
from prompts.analysis_prompts import (
    get_attribute_analysis_prompt,
    get_sentiment_analysis_prompt,
    get_combined_analysis_prompt
)

# OpenAIクライアントの初期化
client = None
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)


class RateLimitMonitor:
    """レート制限を監視するクラス（OpenAIのレート制限に対応）"""
    def __init__(self, max_requests_per_minute: int = 480):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """必要に応じて待機（レート制限を超えないように）"""
        with self.lock:
            now = time.time()
            # 1分以上前のリクエストを削除
            while self.request_times and (now - self.request_times[0]) > 60:
                self.request_times.popleft()
            
            # レート制限の90%に達している場合は待機を検討
            # ただし、レート制限に達する前に次のリクエストを送信できるように最適化
            current_count = len(self.request_times)
            threshold = int(self.max_requests * 0.95)  # 95%の閾値を使用（より積極的に）
            
            if current_count >= threshold:
                # 最も古いリクエストが1分以内にある場合、そのリクエストが1分経過するまで待機
                if self.request_times:
                    oldest_time = self.request_times[0]
                    elapsed = now - oldest_time
                    if elapsed < 60:
                        # 1分経過するまでの残り時間 + 小さなマージン（0.1秒）
                        wait_time = 60 - elapsed + 0.1
                        if wait_time > 0:
                            time.sleep(wait_time)
                            # 再度クリーンアップ
                            now = time.time()
                            while self.request_times and (now - self.request_times[0]) > 60:
                                self.request_times.popleft()
            
            # 現在のリクエスト時刻を記録
            self.request_times.append(time.time())


# グローバルなレート制限監視インスタンス
_rate_limit_monitor = RateLimitMonitor(max_requests_per_minute=480)


def parse_json_response(response_text: str) -> Optional[Dict]:
    """
    JSON形式のレスポンスをパースする
    
    Args:
        response_text: APIレスポンステキスト
        
    Returns:
        パースされたJSON辞書、失敗した場合はNone
    """
    if not response_text or not isinstance(response_text, str):
        return None
    
    try:
        # まず、コードブロック記号を除去
        cleaned_text = response_text.strip()
        # ```json や ``` を除去
        cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'```\s*', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        json_str = None
        
        # JSONコードブロックを抽出（```json ... ```の形式）
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # コードブロックがない場合、直接JSONを探す
            # まず、cleaned_textからJSONを探す（ネストされたJSONにも対応）
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 元のテキストから探す
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # より単純なパターンで試す（貪欲マッチ）
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
        
        # json_strが定義されていない場合はNoneを返す
        if json_str is None:
            import sys
            print(f"DEBUG [JSONパース] JSON文字列が見つかりません。レスポンス: {repr(response_text[:200])}", file=sys.stderr)
            return None
        
        # JSONをパース
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        import sys
        print(f"DEBUG [JSONパース] JSONデコードエラー: {e}, レスポンス: {repr(response_text[:200])}", file=sys.stderr)
        return None
    except (AttributeError, IndexError, TypeError) as e:
        import sys
        print(f"DEBUG [JSONパース] エラー: {e}, レスポンス: {repr(response_text[:200])}", file=sys.stderr)
        return None
    except Exception as e:
        import sys
        print(f"DEBUG [JSONパース] 予期しないエラー: {e}, レスポンス: {repr(response_text[:200])}", file=sys.stderr)
        return None


def analyze_comment_attribute(comment_text: str, username: str, guest_id=None, user_type=None, user_id=None) -> str:
    """
    コメントの属性を分析
    
    Args:
        comment_text: コメント本文
        username: ユーザー名
        guest_id: ゲストID（数値型または文字列型、オプション）
        user_type: ユーザータイプ（オプション、"moderator"の場合に公式コメント判定）
        user_id: ユーザーID（オプション、値が存在する場合に公式コメント判定）
        
    Returns:
        チャットの属性
    """
    import pandas as pd
    
    # user_typeが"moderator"の場合は自動的に公式コメントとして判定
    if user_type and str(user_type).strip().lower() == "moderator":
        return "公式コメント"
    
    # user_idが存在し、値が空でない場合は公式コメントとして判定
    if user_id is not None:
        try:
            # NaN/Noneチェック
            if pd.notna(user_id):
                user_id_str = str(user_id).strip()
                if user_id_str:  # 空文字列でない
                    return "公式コメント"
        except (ValueError, AttributeError, TypeError):
            pass  # user_idが不正な場合は通常の処理に進む
    
    # 後方互換性のため、guest_idによる判定も残す（将来的に削除予定）
    try:
        if guest_id:
            guest_id_str = str(guest_id).strip()
            if guest_id_str == OFFICIAL_GUEST_ID:
                return "公式コメント"
    except (ValueError, AttributeError):
        pass  # guest_idが不正な場合は通常の処理に進む
    
    # usernameが"マツキヨココカラSTAFF"の場合は公式コメントとして判定
    if username and str(username).strip() == "マツキヨココカラSTAFF":
        return "公式コメント"
    
    prompt = get_attribute_analysis_prompt(comment_text, username)
    
    if not client:
        raise ValueError("OPENAI_API_KEYが設定されていません。")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_completion_tokens=100,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # 一貫性を重視
        )
        
        raw_response = response.choices[0].message.content
        
        # デバッグ情報: 最初の10件のみログ出力
        import sys
        if not hasattr(analyze_comment_attribute, '_debug_count'):
            analyze_comment_attribute._debug_count = 0
        if analyze_comment_attribute._debug_count < 10:
            print(f"DEBUG [属性分析] コメント: {comment_text[:50]}...", file=sys.stderr)
            print(f"DEBUG [属性分析] 生レスポンス: {repr(raw_response)}", file=sys.stderr)
            analyze_comment_attribute._debug_count += 1
        
        # レスポンスから属性名を抽出
        response_cleaned = raw_response.strip()
        response_cleaned = re.sub(r'\s+', ' ', response_cleaned)
        response_cleaned = response_cleaned.strip()
        
        # まず、レスポンス全体から直接カテゴリ名を検索
        attribute = None
        matched_attribute = None
        response_lower = response_cleaned.lower()
        
        for attr in CHAT_ATTRIBUTES:
            attr_lower = attr.lower()
            # 完全一致（大文字小文字を区別しない）
            if attr_lower == response_lower or attr == response_cleaned:
                matched_attribute = attr
                break
            # 部分一致（カテゴリ名がレスポンスに含まれている）
            if attr in response_cleaned or attr_lower in response_lower:
                matched_attribute = attr
                break
        
        if matched_attribute:
            attribute = matched_attribute
            if analyze_comment_attribute._debug_count <= 10:
                print(f"DEBUG [属性分析] マッチ成功: {matched_attribute}", file=sys.stderr)
        else:
            # マッチしない場合は生レスポンスをそのまま使用
            attribute = response_cleaned
            if analyze_comment_attribute._debug_count <= 10:
                print(f"DEBUG [属性分析] マッチなし、生レスポンスを使用: {repr(attribute)}", file=sys.stderr)
        
        # レスポンスから余分な文字を除去（改行、空白、句読点など）
        # 改行、タブ、連続する空白を1つの空白に変換
        attribute_cleaned = re.sub(r'\s+', ' ', attribute)
        # 前後の空白を除去
        attribute_cleaned = attribute_cleaned.strip()
        # 句読点を除去
        attribute_cleaned = attribute_cleaned.replace('。', '').replace('、', '').replace('.', '').replace(',', '')
        
        # 取得した属性がリストに含まれているか確認（完全一致）
        if attribute_cleaned in CHAT_ATTRIBUTES:
            # 「公式コメント」が返された場合、usernameをチェック
            if attribute_cleaned == "公式コメント":
                # usernameがStarbucks Coffee Japanの場合のみ「公式コメント」を返す
                if username == COMPANY_NAME:
                    if analyze_comment_attribute._debug_count <= 10:
                        print(f"DEBUG [属性分析] 「公式コメント」が返され、usernameが{COMPANY_NAME}のため「公式コメント」にマッピング", file=sys.stderr)
                    return "公式コメント"
                else:
                    # usernameがStarbucks Coffee Japanでない場合は「絵文字のみ」にマッピング
                    if analyze_comment_attribute._debug_count <= 10:
                        print(f"DEBUG [属性分析] 「公式コメント」が返されたが、usernameが{COMPANY_NAME}ではないため「絵文字のみ」にマッピング", file=sys.stderr)
                    return "絵文字のみ"
            if analyze_comment_attribute._debug_count <= 10:
                print(f"DEBUG [属性分析] 完全一致でマッチ: {attribute_cleaned}", file=sys.stderr)
            return attribute_cleaned
        
        # 元のattributeでも確認
        if attribute in CHAT_ATTRIBUTES:
            # 「公式コメント」が返された場合、usernameをチェック
            if attribute == "公式コメント":
                # usernameがStarbucks Coffee Japanの場合のみ「公式コメント」を返す
                if username == COMPANY_NAME:
                    if analyze_comment_attribute._debug_count <= 10:
                        print(f"DEBUG [属性分析] 「公式コメント」が返され、usernameが{COMPANY_NAME}のため「公式コメント」にマッピング", file=sys.stderr)
                    return "公式コメント"
                else:
                    # usernameがStarbucks Coffee Japanでない場合は「絵文字のみ」にマッピング
                    if analyze_comment_attribute._debug_count <= 10:
                        print(f"DEBUG [属性分析] 「公式コメント」が返されたが、usernameが{COMPANY_NAME}ではないため「絵文字のみ」にマッピング", file=sys.stderr)
                    return "絵文字のみ"
            if analyze_comment_attribute._debug_count <= 10:
                print(f"DEBUG [属性分析] 完全一致でマッチ（元の値）: {attribute}", file=sys.stderr)
            return attribute
        
        # 部分一致で検索（大文字小文字を区別しない）
        attribute_lower = attribute_cleaned.lower()
        matched_attribute = None
        for attr in CHAT_ATTRIBUTES:
            attr_lower = attr.lower()
            # 完全一致（大文字小文字を区別しない）
            if attr_lower == attribute_lower:
                matched_attribute = attr
                break
            # 部分一致
            if attr in attribute_cleaned or attribute_cleaned in attr:
                matched_attribute = attr
                break
            # 大文字小文字を無視した部分一致
            if attr_lower in attribute_lower or attribute_lower in attr_lower:
                matched_attribute = attr
                break
        
        # マッチした属性がある場合
        if matched_attribute:
            # 「公式コメント」が返された場合、usernameをチェック
            if matched_attribute == "公式コメント":
                # usernameがStarbucks Coffee Japanの場合のみ「公式コメント」を返す
                if username == COMPANY_NAME:
                    if analyze_comment_attribute._debug_count <= 10:
                        print(f"DEBUG [属性分析] 「公式コメント」が返され、usernameが{COMPANY_NAME}のため「公式コメント」にマッピング", file=sys.stderr)
                    return "公式コメント"
                else:
                    # usernameがStarbucks Coffee Japanでない場合は「絵文字のみ」にマッピング
                    if analyze_comment_attribute._debug_count <= 10:
                        print(f"DEBUG [属性分析] 「公式コメント」が返されたが、usernameが{COMPANY_NAME}ではないため「絵文字のみ」にマッピング", file=sys.stderr)
                    return "絵文字のみ"
            if analyze_comment_attribute._debug_count <= 10:
                print(f"DEBUG [属性分析] マッチした属性: {matched_attribute}", file=sys.stderr)
            return matched_attribute
        
        # 見つからない場合は「その他」を返す
        if analyze_comment_attribute._debug_count <= 10:
            print(f"DEBUG [属性分析] マッチなし、デフォルト値を返す。元のレスポンス: {repr(raw_response)}", file=sys.stderr)
            print(f"DEBUG [属性分析] 利用可能なカテゴリ: {CHAT_ATTRIBUTES}", file=sys.stderr)
        
        # 最終的な検証：返す値が有効なカテゴリに含まれているか確認
        final_attribute = "絵文字のみ"
        if final_attribute not in CHAT_ATTRIBUTES:
            import sys
            if analyze_comment_attribute._debug_count <= 10:
                print("DEBUG [属性分析] 警告: デフォルト値が無効です。最初のカテゴリを使用します。", file=sys.stderr)
            final_attribute = CHAT_ATTRIBUTES[0] if CHAT_ATTRIBUTES else "絵文字のみ"
        
        return final_attribute
        
    except Exception as e:
        import sys
        print(f"属性分析エラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return "絵文字のみ"


def analyze_comment_sentiment(comment_text: str) -> str:
    """
    コメントの感情を分析
    
    Args:
        comment_text: コメント本文
        
    Returns:
        チャット感情
    """
    prompt = get_sentiment_analysis_prompt(comment_text)
    
    if not client:
        raise ValueError("OPENAI_API_KEYが設定されていません。")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_completion_tokens=100,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # 一貫性を重視
        )
        
        raw_response = response.choices[0].message.content
        
        # デバッグ情報: 最初の10件のみログ出力
        import sys
        if not hasattr(analyze_comment_sentiment, '_debug_count'):
            analyze_comment_sentiment._debug_count = 0
        if analyze_comment_sentiment._debug_count < 10:
            print(f"DEBUG [感情分析] コメント: {comment_text[:50]}...", file=sys.stderr)
            print(f"DEBUG [感情分析] 生レスポンス: {repr(raw_response)}", file=sys.stderr)
            analyze_comment_sentiment._debug_count += 1
        
        # レスポンスから感情名を抽出
        response_cleaned = raw_response.strip()
        response_cleaned = re.sub(r'\s+', ' ', response_cleaned)
        response_cleaned = response_cleaned.strip()
        
        # まず、レスポンス全体から直接カテゴリ名を検索
        sentiment = None
        matched_sentiment = None
        response_lower = response_cleaned.lower()
        
        for sent in CHAT_SENTIMENTS:
            sent_lower = sent.lower()
            # 完全一致（大文字小文字を区別しない）
            if sent_lower == response_lower or sent == response_cleaned:
                matched_sentiment = sent
                break
            # 部分一致（カテゴリ名がレスポンスに含まれている）
            if sent in response_cleaned or sent_lower in response_lower:
                matched_sentiment = sent
                break
        
        if matched_sentiment:
            sentiment = matched_sentiment
            if analyze_comment_sentiment._debug_count <= 10:
                print(f"DEBUG [感情分析] マッチ成功: {matched_sentiment}", file=sys.stderr)
        else:
            # マッチしない場合は生レスポンスをそのまま使用
            sentiment = response_cleaned
            if analyze_comment_sentiment._debug_count <= 10:
                print(f"DEBUG [感情分析] マッチなし、生レスポンスを使用: {repr(sentiment)}", file=sys.stderr)
        
        # レスポンスから余分な文字を除去（改行、空白、句読点など）
        # 改行、タブ、連続する空白を1つの空白に変換
        sentiment_cleaned = re.sub(r'\s+', ' ', sentiment)
        # 前後の空白を除去
        sentiment_cleaned = sentiment_cleaned.strip()
        # 句読点を除去
        sentiment_cleaned = sentiment_cleaned.replace('。', '').replace('、', '').replace('.', '').replace(',', '')
        
        # 取得した感情がリストに含まれているか確認（完全一致）
        if sentiment_cleaned in CHAT_SENTIMENTS:
            if analyze_comment_sentiment._debug_count <= 10:
                print(f"DEBUG [感情分析] 完全一致でマッチ: {sentiment_cleaned}", file=sys.stderr)
            return sentiment_cleaned
        
        # 元のsentimentでも確認
        if sentiment in CHAT_SENTIMENTS:
            if analyze_comment_sentiment._debug_count <= 10:
                print(f"DEBUG [感情分析] 完全一致でマッチ（元の値）: {sentiment}", file=sys.stderr)
            return sentiment
        
        # 部分一致で検索（大文字小文字を区別しない）
        sentiment_lower = sentiment_cleaned.lower()
        for sent in CHAT_SENTIMENTS:
            sent_lower = sent.lower()
            # 完全一致（大文字小文字を区別しない）
            if sent_lower == sentiment_lower:
                if analyze_comment_sentiment._debug_count <= 10:
                    print(f"DEBUG [感情分析] 大文字小文字を無視して完全一致: {sent}", file=sys.stderr)
                return sent
            # 部分一致
            if sent in sentiment_cleaned or sentiment_cleaned in sent:
                if analyze_comment_sentiment._debug_count <= 10:
                    print(f"DEBUG [感情分析] 部分一致でマッチ: {sent}", file=sys.stderr)
                return sent
            # 大文字小文字を無視した部分一致
            if sent_lower in sentiment_lower or sentiment_lower in sent_lower:
                if analyze_comment_sentiment._debug_count <= 10:
                    print(f"DEBUG [感情分析] 大文字小文字を無視して部分一致: {sent}", file=sys.stderr)
                return sent
        
        # 見つからない場合は「どちらでもない」を返す
        if analyze_comment_sentiment._debug_count <= 10:
            print(f"DEBUG [感情分析] マッチなし、デフォルト値を返す。元のレスポンス: {repr(raw_response)}", file=sys.stderr)
            print(f"DEBUG [感情分析] 利用可能なカテゴリ: {CHAT_SENTIMENTS}", file=sys.stderr)
        
        # 最終的な検証：返す値が有効なカテゴリに含まれているか確認
        final_sentiment = "どちらでもない"
        if final_sentiment not in CHAT_SENTIMENTS:
            import sys
            if analyze_comment_sentiment._debug_count <= 10:
                print("DEBUG [感情分析] 警告: デフォルト値が無効です。最初のカテゴリを使用します。", file=sys.stderr)
            final_sentiment = CHAT_SENTIMENTS[0] if CHAT_SENTIMENTS else "どちらでもない"
        
        return final_sentiment
        
    except Exception as e:
        import sys
        print(f"感情分析エラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return "どちらでもない"


def analyze_comment_combined(comment_text: str, username: str, guest_id=None, user_type=None, user_id=None) -> tuple:
    """
    コメントの属性と感情を1回のAPI呼び出しで分析（高速化版）
    
    Args:
        comment_text: コメント本文
        username: ユーザー名
        guest_id: ゲストID（数値型または文字列型、オプション）
        user_type: ユーザータイプ（オプション、"moderator"の場合に公式コメント判定）
        user_id: ユーザーID（オプション、値が存在する場合に公式コメント判定）
        
    Returns:
        (attribute, sentiment, tokens_info) のタプル
        tokens_infoは {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int} の辞書
    """
    import pandas as pd
    
    # user_typeが"moderator"の場合は自動的に公式コメントとして判定
    if user_type and str(user_type).strip().lower() == "moderator":
        # 感情だけ分析
        sentiment = analyze_comment_sentiment(comment_text)
        return ("公式コメント", sentiment)
    
    # user_idが存在し、値が空でない場合は公式コメントとして判定
    if user_id is not None:
        try:
            # NaN/Noneチェック
            if pd.notna(user_id):
                user_id_str = str(user_id).strip()
                if user_id_str:  # 空文字列でない
                    # 感情だけ分析（トークン使用量は0として返す）
                    sentiment = analyze_comment_sentiment(comment_text)
                    return ("公式コメント", sentiment, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        except (ValueError, AttributeError, TypeError):
            pass  # user_idが不正な場合は通常の処理に進む
    
    # 後方互換性のため、guest_idによる判定も残す（将来的に削除予定）
    try:
        if guest_id:
            guest_id_str = str(guest_id).strip()
            if guest_id_str == OFFICIAL_GUEST_ID:
                # 感情だけ分析（トークン使用量は0として返す）
                sentiment = analyze_comment_sentiment(comment_text)
                return ("公式コメント", sentiment, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    except (ValueError, AttributeError):
        pass  # guest_idが不正な場合は通常の処理に進む
    
    # usernameが"マツキヨココカラSTAFF"の場合は公式コメントとして判定
    if username and str(username).strip() == "マツキヨココカラSTAFF":
        # 感情だけ分析（トークン使用量は0として返す）
        sentiment = analyze_comment_sentiment(comment_text)
        return ("公式コメント", sentiment, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    
    prompt = get_combined_analysis_prompt(comment_text, username)
    
    if not client:
        raise ValueError("OPENAI_API_KEYが設定されていません。")
    
    try:
        # レート制限エラー対応：最大3回リトライ
        max_retries = 3
        retry_delay = 60  # 60秒待機
        
        api_response = None
        for attempt in range(max_retries):
            try:
                api_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_completion_tokens=200,  # 統合プロンプトなので少し増やす
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1  # 一貫性を重視
                )
                break  # 成功したらループを抜ける
                
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"レート制限エラー (試行 {attempt + 1}/{max_retries}): {e}. {retry_delay}秒待機してリトライします...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数バックオフ
                else:
                    # 最大リトライ回数に達した場合
                    error_msg = "レート制限エラー: 最大リトライ回数に達しました。結果を保存して処理を中断します。"
                    print(error_msg)
                    raise Exception(error_msg)
            except openai.APIError as e:
                # 429エラー（レート制限）を特別に処理
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        print(f"レート制限エラー (試行 {attempt + 1}/{max_retries}): {e}. {retry_delay}秒待機してリトライします...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        error_msg = f"レート制限エラーが発生しました。処理を中断して、続きから再開してください。エラー詳細: {error_str}"
                        print(error_msg)
                        raise Exception(error_msg)
                else:
                    raise
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                # ネットワークエラーやタイムアウトエラー
                error_str_conn = str(e)
                if attempt < max_retries - 1:
                    print(f"接続エラー (試行 {attempt + 1}/{max_retries}): {e}. {retry_delay}秒待機してリトライします...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    error_msg = f"接続エラーが発生しました。処理を中断して、続きから再開してください。エラー詳細: {error_str_conn}"
                    print(error_msg)
                    raise Exception(error_msg)
        
        if api_response is None:
            raise Exception("API呼び出しに失敗しました。")
        
        # トークン使用量を取得
        tokens_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(api_response, 'usage') and api_response.usage:
            usage = api_response.usage
            tokens_info = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
        
        raw_response = api_response.choices[0].message.content
        
        # デバッグ情報: 最初の10件のみログ出力
        import sys
        if not hasattr(analyze_comment_combined, '_debug_count'):
            analyze_comment_combined._debug_count = 0
        if analyze_comment_combined._debug_count < 10:
            print(f"DEBUG [統合分析] コメント: {comment_text[:50]}...", file=sys.stderr)
            print(f"DEBUG [統合分析] 生レスポンス: {repr(raw_response)}", file=sys.stderr)
            analyze_comment_combined._debug_count += 1
        
        # レスポンスから属性と感情を抽出（行単位で解析）
        attribute = None
        sentiment = None
        response = raw_response.strip()
        
        # 行単位で解析（「属性:」と「感情:」の形式を想定）
        lines = response.split('\n')
        for line in lines:
            line_cleaned = re.sub(r'\s+', ' ', line.strip())
            line_cleaned = line_cleaned.strip()
            
            # 属性の抽出（「属性:」や「属性：」のパターンに対応）
            if not attribute and '属性' in line_cleaned:
                for sep in [':', '：', '=']:
                    if sep in line_cleaned:
                        parts = line_cleaned.split(sep, 1)
                        if len(parts) > 1:
                            attr_text = parts[1].strip()
                            attr_text = attr_text.replace('。', '').replace('、', '').replace('.', '').replace(',', '')
                            attr_text_lower = attr_text.lower()
                            
                            # 属性を検証
                            for attr in CHAT_ATTRIBUTES:
                                if attr.lower() == attr_text_lower or attr in attr_text or attr_text in attr:
                                    # 「公式コメント」が返された場合、usernameをチェック
                                    if attr == "公式コメント":
                                        if username == COMPANY_NAME:
                                            attribute = "公式コメント"
                                        else:
                                            attribute = "絵文字のみ"
                                    else:
                                        attribute = attr
                                    break
                            if attribute:
                                break
            
            # 感情の抽出（「感情:」や「感情：」のパターンに対応）
            if not sentiment and '感情' in line_cleaned:
                for sep in [':', '：', '=']:
                    if sep in line_cleaned:
                        parts = line_cleaned.split(sep, 1)
                        if len(parts) > 1:
                            sent_text = parts[1].strip()
                            sent_text = sent_text.replace('。', '').replace('、', '').replace('.', '').replace(',', '')
                            sent_text_lower = sent_text.lower()
                            
                            # 感情を検証
                            for sent in CHAT_SENTIMENTS:
                                if sent.lower() == sent_text_lower or sent in sent_text or sent_text in sent:
                                    sentiment = sent
                                    break
                            if sentiment:
                                break
            
            if attribute and sentiment:
                break
        
        # 行単位のパースで取得できなかった場合、レスポンス全体から検索
        if not attribute or not sentiment:
            response_cleaned = re.sub(r'\s+', ' ', response)
            response_cleaned = response_cleaned.strip()
            response_lower = response_cleaned.lower()
            
            # 属性の検索
            if not attribute:
                for attr in CHAT_ATTRIBUTES:
                    attr_lower = attr.lower()
                    # 完全一致（大文字小文字を区別しない）
                    if attr_lower == response_lower or attr == response_cleaned:
                        # 「公式コメント」が返された場合、usernameをチェック
                        if attr == "公式コメント":
                            if username == COMPANY_NAME:
                                attribute = "公式コメント"
                            else:
                                attribute = "絵文字のみ"
                        else:
                            attribute = attr
                        if attribute:
                            break
                    # 部分一致
                    if attr in response_cleaned or response_cleaned in attr:
                        if attr == "公式コメント":
                            if username == COMPANY_NAME:
                                attribute = "公式コメント"
                            else:
                                attribute = "絵文字のみ"
                        else:
                            attribute = attr
                        if attribute:
                            break
                    # 大文字小文字を無視した部分一致
                    if attr_lower in response_lower or response_lower in attr_lower:
                        if attr == "公式コメント":
                            if username == COMPANY_NAME:
                                attribute = "公式コメント"
                            else:
                                attribute = "絵文字のみ"
                        else:
                            attribute = attr
                        if attribute:
                            break
            
            # 感情の検索
            if not sentiment:
                for sent in CHAT_SENTIMENTS:
                    sent_lower = sent.lower()
                    if sent_lower == response_lower or sent == response_cleaned:
                        sentiment = sent
                        break
                    if sent in response_cleaned or response_cleaned in sent:
                        sentiment = sent
                        break
                    if sent_lower in response_lower or response_lower in sent_lower:
                        sentiment = sent
                        break
                    if sentiment:
                        break
        if analyze_comment_combined._debug_count <= 10:
            print(f"DEBUG [統合分析] パース結果 - 属性: {attribute}, 感情: {sentiment}", file=sys.stderr)
        
        # パースに失敗した場合のフォールバック
        if not attribute:
            if analyze_comment_combined._debug_count <= 10:
                print("DEBUG [統合分析] 属性のパース失敗、フォールバック処理を実行", file=sys.stderr)
            # 従来の方法で属性を取得
            user_type = None  # user_typeは利用できない場合がある
            user_id = None  # user_idは利用できない場合がある
            attribute = analyze_comment_attribute(comment_text, username, guest_id, user_type, user_id)
        if not sentiment:
            if analyze_comment_combined._debug_count <= 10:
                print("DEBUG [統合分析] 感情のパース失敗、フォールバック処理を実行", file=sys.stderr)
            # 従来の方法で感情を取得
            sentiment = analyze_comment_sentiment(comment_text)
        
        # 最終的なattributeが「公式コメント」の場合、usernameをチェック
        if attribute == "公式コメント":
            # usernameがStarbucks Coffee Japanの場合のみ「公式コメント」を返す
            if username == COMPANY_NAME:
                if analyze_comment_combined._debug_count <= 10:
                    print(f"DEBUG [統合分析] 最終チェック: 「公式コメント」が返され、usernameが{COMPANY_NAME}のため「公式コメント」にマッピング", file=sys.stderr)
                attribute = "公式コメント"
            else:
                # usernameがStarbucks Coffee Japanでない場合は「絵文字のみ」にマッピング
                if analyze_comment_combined._debug_count <= 10:
                    print(f"DEBUG [統合分析] 最終チェック: 「公式コメント」が返されたが、usernameが{COMPANY_NAME}ではないため「絵文字のみ」にマッピング", file=sys.stderr)
                attribute = "絵文字のみ"
        
        # 最終的な検証：属性と感情が有効なカテゴリに含まれているか確認
        if attribute not in CHAT_ATTRIBUTES:
            import sys
            if analyze_comment_combined._debug_count <= 10:
                print(f"DEBUG [統合分析] 警告: 無効な属性が返されました: {repr(attribute)}。レスポンス全体から再検索します。", file=sys.stderr)
                print(f"DEBUG [統合分析] 生レスポンス: {repr(raw_response[:500])}", file=sys.stderr)
            
            # 無効な属性の場合、レスポンス全体から再検索
            response_cleaned = raw_response.strip()
            response_cleaned = re.sub(r'\s+', ' ', response_cleaned)
            response_cleaned = response_cleaned.strip()
            response_lower = response_cleaned.lower()
            
            # レスポンス全体からカテゴリ名を検索
            matched_attribute = None
            for attr in CHAT_ATTRIBUTES:
                attr_lower = attr.lower()
                # 部分一致で検索（より積極的に）
                if attr in response_cleaned or attr_lower in response_lower or response_cleaned in attr or response_lower in attr_lower:
                    matched_attribute = attr
                    if analyze_comment_combined._debug_count <= 10:
                        print(f"DEBUG [統合分析] 再検索でマッチ: {matched_attribute}", file=sys.stderr)
                    break
            
            if matched_attribute:
                attribute = matched_attribute
                # 「公式コメント」の場合はusernameをチェック
                if attribute == "公式コメント":
                    if username == COMPANY_NAME:
                        attribute = "公式コメント"
                    else:
                        attribute = "絵文字のみ"
            else:
                # マッチしない場合のみデフォルト値を使用
                if analyze_comment_combined._debug_count <= 10:
                    print(f"DEBUG [統合分析] 再検索でもマッチせず、デフォルト値を使用: 絵文字のみ", file=sys.stderr)
                attribute = "絵文字のみ"
        
        if sentiment not in CHAT_SENTIMENTS:
            import sys
            if analyze_comment_combined._debug_count <= 10:
                print(f"DEBUG [統合分析] 警告: 無効な感情が返されました: {repr(sentiment)}。レスポンス全体から再検索します。", file=sys.stderr)
                print(f"DEBUG [統合分析] 生レスポンス: {repr(raw_response[:500])}", file=sys.stderr)
            
            # 無効な感情の場合、レスポンス全体から再検索
            response_cleaned = raw_response.strip()
            response_cleaned = re.sub(r'\s+', ' ', response_cleaned)
            response_cleaned = response_cleaned.strip()
            response_lower = response_cleaned.lower()
            
            # レスポンス全体からカテゴリ名を検索
            matched_sentiment = None
            for sent in CHAT_SENTIMENTS:
                sent_lower = sent.lower()
                # 部分一致で検索（より積極的に）
                if sent in response_cleaned or sent_lower in response_lower or response_cleaned in sent or response_lower in sent_lower:
                    matched_sentiment = sent
                    if analyze_comment_combined._debug_count <= 10:
                        print(f"DEBUG [統合分析] 再検索でマッチ: {matched_sentiment}", file=sys.stderr)
                    break
            
            if matched_sentiment:
                sentiment = matched_sentiment
            else:
                # マッチしない場合のみデフォルト値を使用
                if analyze_comment_combined._debug_count <= 10:
                    print(f"DEBUG [統合分析] 再検索でもマッチせず、デフォルト値を使用: どちらでもない", file=sys.stderr)
                sentiment = "どちらでもない"
        
        if analyze_comment_combined._debug_count <= 10:
            print(f"DEBUG [統合分析] 最終結果 - 属性: {attribute}, 感情: {sentiment}", file=sys.stderr)
        
        # トークン使用量を取得
        if api_response and hasattr(api_response, 'usage') and api_response.usage:
            usage = api_response.usage
            tokens_info = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
        else:
            tokens_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        return (attribute, sentiment, tokens_info)
        
    except Exception as e:
        import sys
        import traceback
        error_str = str(e)
        error_type = type(e).__name__
        
        # デバッグ情報: エラーの詳細を出力
        print(f"DEBUG [統合分析] エラー発生 - タイプ: {error_type}, メッセージ: {error_str}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
        # レート制限エラーの場合は例外を再発生（再開機能を使えるようにする）
        if "レート制限" in error_str or "rate_limit" in error_str.lower() or "429" in error_str:
            print(f"統合分析エラー（レート制限）: {e}", file=sys.stderr)
            raise Exception(f"レート制限エラーが発生しました。処理を中断して、続きから再開してください。エラー詳細: {error_str}")
        else:
            print(f"統合分析エラー: {e}", file=sys.stderr)
            # エラー時は従来の方法にフォールバック
            try:
                print("DEBUG [統合分析] フォールバック処理を実行", file=sys.stderr)
                attribute = analyze_comment_attribute(comment_text, username, guest_id, None, None)
                sentiment = analyze_comment_sentiment(comment_text)
                print(f"DEBUG [統合分析] フォールバック成功 - 属性: {attribute}, 感情: {sentiment}", file=sys.stderr)
                tokens_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                return (attribute, sentiment, tokens_info)
            except Exception as fallback_error:
                # フォールバックも失敗した場合
                error_str_fallback = str(fallback_error)
                error_type_fallback = type(fallback_error).__name__
                print(f"DEBUG [統合分析] フォールバックも失敗 - タイプ: {error_type_fallback}, メッセージ: {error_str_fallback}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                
                if "レート制限" in error_str_fallback or "rate_limit" in error_str_fallback.lower() or "429" in error_str_fallback:
                    raise Exception(f"レート制限エラーが発生しました。処理を中断して、続きから再開してください。エラー詳細: {error_str_fallback}")
                # その他のエラーはデフォルト値を返す
                print("DEBUG [統合分析] デフォルト値を返す", file=sys.stderr)
                return ("絵文字のみ", "どちらでもない")


def _analyze_single_comment(idx, row, rate_limit_monitor):
    """
    1つのコメントを分析するヘルパー関数（並列処理用）
    
    Args:
        idx: インデックス
        row: データフレームの行
        rate_limit_monitor: レート制限監視インスタンス
        
    Returns:
        (idx, result_dict) のタプル。エラー時は (idx, None)
    """
    comment_text = str(row["original_text"])
    username = str(row["username"])
    guest_id = row.get("guest_id", None)
    user_type = row.get("user_type", None)
    user_id = row.get("user_id", None)
    
    # レート制限をチェックしてから処理
    rate_limit_monitor.wait_if_needed()
    
    try:
        # 統合分析（1回のAPI呼び出しで属性と感情の両方を取得：50%高速化）
        attribute, sentiment, tokens_info = analyze_comment_combined(comment_text, username, guest_id, user_type, user_id)
        
        # 結果を保存
        result_row = row.to_dict()
        result_row["チャットの属性"] = attribute
        result_row["チャット感情"] = sentiment
        result_row["_tokens_info"] = tokens_info  # トークン情報を一時的に保存
        
        return (idx, result_row)
    except Exception as e:
        error_str = str(e)
        # レート制限エラーは再発生させる
        if "レート制限" in error_str or "rate_limit" in error_str.lower() or "429" in error_str:
            raise  # 上位で処理
        # その他のエラーはデフォルト値を返す
        result_row = row.to_dict()
        result_row["チャットの属性"] = "絵文字のみ"
        result_row["チャット感情"] = "どちらでもない"
        return (idx, result_row)


def analyze_all_comments(df, progress_callback=None, save_callback=None, check_cancel_callback=None) -> Dict:
    """
    全コメントを分析（高速化版：統合プロンプト使用）
    
    Args:
        df: データフレーム
        progress_callback: 進捗コールバック関数
        save_callback: 中間結果保存コールバック関数（PCスリープ対策）
        check_cancel_callback: 中断チェックコールバック関数（Noneを返すかFalseを返すと継続、Trueを返すと中断）
        
    Returns:
        分析結果を含むデータフレーム
        
    Raises:
        KeyboardInterrupt: 中断がリクエストされた場合
    """
    import pandas as pd
    
    total = len(df)
    results = []
    
    # 既存の結果がある場合は読み込む（再開機能）
    start_idx = 0
    if save_callback:
        existing_results = save_callback("load")
        if existing_results and len(existing_results) > 0:
            # 保存された結果がDataFrameの場合はリストに変換
            if isinstance(existing_results, pd.DataFrame):
                results = existing_results.to_dict('records')
            elif isinstance(existing_results, list):
                results = existing_results
            else:
                results = []
            
            start_idx = len(results)
            if progress_callback:
                progress_callback(start_idx, total)
    
    # 並列処理の設定
    max_workers = 8  # 8コメントを同時処理（50%高速化）
    batch_size = max_workers  # バッチサイズ
    results_dict = {}  # インデックスをキーとした結果辞書
    results_lock = threading.Lock()  # 結果のスレッドセーフなアクセス用
    
    # レート制限監視インスタンス（OpenAIのレート制限に合わせて480リクエスト/分に設定）
    rate_monitor = RateLimitMonitor(max_requests_per_minute=480)
    
    # レート制限エラーの監視用カウンター（テスト用）
    rate_limit_error_count = 0
    
    # トークン使用量の累積
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    
    # バッチ処理で並列実行
    completed_count = start_idx
    idx = start_idx
    
    while idx < total:
        # 中断チェック
        if check_cancel_callback and check_cancel_callback():
            if save_callback:
                sorted_results = [results_dict.get(i) for i in sorted(results_dict.keys()) if results_dict.get(i) is not None]
                save_callback("save", sorted_results)
            raise KeyboardInterrupt("分析がユーザーによって中断されました。")
        
        # バッチの準備（最大8コメント）
        batch_indices = []
        batch_rows = []
        for _ in range(batch_size):
            if idx >= total:
                break
            batch_indices.append(idx)
            batch_rows.append(df.iloc[idx])
            idx += 1
        
        if not batch_indices:
            break
        
        # バッチを並列処理
        with ThreadPoolExecutor(max_workers=len(batch_indices)) as executor:
            futures = {}
            for batch_idx, batch_row in zip(batch_indices, batch_rows):
                future = executor.submit(_analyze_single_comment, batch_idx, batch_row, rate_monitor)
                futures[future] = batch_idx
            
            # 完了したタスクから結果を取得
            for future in as_completed(futures):
                # 中断チェック（並列処理中）
                if check_cancel_callback and check_cancel_callback():
                    # 残りのタスクをキャンセル
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    if save_callback:
                        sorted_results = [results_dict.get(i) for i in sorted(results_dict.keys()) if results_dict.get(i) is not None]
                        save_callback("save", sorted_results)
                    raise KeyboardInterrupt("分析がユーザーによって中断されました。")
                
                batch_idx = futures[future]
                try:
                    _, result_row = future.result()
                    with results_lock:
                        # トークン使用量を累積
                        if "_tokens_info" in result_row:
                            tokens_info = result_row.pop("_tokens_info")  # トークン情報を取得して削除
                            total_prompt_tokens += tokens_info.get("prompt_tokens", 0)
                            total_completion_tokens += tokens_info.get("completion_tokens", 0)
                            total_tokens += tokens_info.get("total_tokens", 0)
                        
                        results_dict[batch_idx] = result_row
                        completed_count += 1
                        
                        # 進捗更新
                        if progress_callback:
                            progress_callback(completed_count, total)
                        
                        # 中間結果を保存（PCスリープ対策：10件ごとに保存）
                        if save_callback and completed_count % 10 == 0:
                            sorted_results = [results_dict.get(i) for i in sorted(results_dict.keys()) if results_dict.get(i) is not None]
                            save_callback("save", sorted_results)
                            
                except Exception as e:
                    # レート制限エラーなどの場合
                    error_str = str(e)
                    if "レート制限" in error_str or "rate_limit" in error_str.lower() or "429" in error_str:
                        # レート制限エラーのカウント（テスト用）
                        rate_limit_error_count += 1
                        import sys
                        print(f"警告: レート制限エラーが発生しました（{rate_limit_error_count}回目）: {error_str}", file=sys.stderr)
                        # 現在の結果を保存
                        if save_callback:
                            sorted_results = [results_dict.get(i) for i in sorted(results_dict.keys()) if results_dict.get(i) is not None]
                            save_callback("save", sorted_results)
                        # エラーを再発生（上位で処理）
                        raise Exception(f"レート制限エラーが発生しました。処理を中断して、続きから再開してください。エラー詳細: {error_str}")
                    else:
                        print(f"分析エラー (インデックス {batch_idx}): {e}")
                        # エラーが発生したコメントはデフォルト値で処理
                        row = df.iloc[batch_idx]
                        result_row = row.to_dict()
                        result_row["チャットの属性"] = "絵文字のみ"
                        result_row["チャット感情"] = "どちらでもない"
                        with results_lock:
                            results_dict[batch_idx] = result_row
                            completed_count += 1
                            if progress_callback:
                                progress_callback(completed_count, total)
        
        # バッチ間で少し待機（レート制限対策）
        if idx < total:
            time.sleep(0.1)
    
    # 結果をインデックス順にソート
    sorted_indices = sorted(results_dict.keys())
    results = [results_dict[i] for i in sorted_indices if results_dict[i] is not None]
    
    # 最終結果を保存
    if save_callback:
        save_callback("save", results)
        save_callback("clear")  # 一時ファイルをクリア
    
    # レート制限エラーの監視結果を出力（テスト用）
    import sys
    if rate_limit_error_count > 0:
        print(f"\n警告: レート制限エラーが{rate_limit_error_count}回発生しました。", file=sys.stderr)
    else:
        print("\n✓ レート制限エラーは発生しませんでした（8並列処理で正常に動作しました）。", file=sys.stderr)
    
    result_df = pd.DataFrame(results)
    
    # トークン使用量情報を返り値に含める
    return {
        "df": result_df,
        "api_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens
        }
    }

