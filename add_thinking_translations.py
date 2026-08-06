#!/usr/bin/env python3
"""Add thinking/reasoning translations to all .ts files."""

import re
import glob
import os

# Translations for each language
# Format: {lang_code: {source_string: translation}}
TRANSLATIONS = {
    "de": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "Denken oder Reasoning (nur OpenAI-kompatibel)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* Betrifft nur LLM-Übersetzer, die dies unterstützen. Gilt nicht für DeepL / Microsoft / Google.",
        "Off": "Aus",
        "Low": "Niedrig",
        "Medium": "Mittel",
        "High": "Hoch",
    },
    "es": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "Pensamiento o Razonamiento (solo compatible con OpenAI)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* Solo afecta a traductores LLM que lo admiten. No se aplica a DeepL / Microsoft / Google.",
        "Off": "Desactivado",
        "Low": "Bajo",
        "Medium": "Medio",
        "High": "Alto",
    },
    "fr": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "Réflexion ou Raisonnement (compatible OpenAI uniquement)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* N'affecte que les traducteurs LLM qui le prennent en charge. Ne s'applique pas à DeepL / Microsoft / Google.",
        "Off": "Désactivé",
        "Low": "Faible",
        "Medium": "Moyen",
        "High": "Élevé",
    },
    "it": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "Pensiero o Ragionamento (solo compatibile con OpenAI)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* Influisce solo sui traduttori LLM che lo supportano. Non si applica a DeepL / Microsoft / Google.",
        "Off": "Disattivato",
        "Low": "Basso",
        "Medium": "Medio",
        "High": "Alto",
    },
    "ja": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "思考または推論 (OpenAI互換のみ)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* 対応しているLLM翻訳者のみに影響します。DeepL / Microsoft / Googleには適用されません。",
        "Off": "オフ",
        "Low": "低",
        "Medium": "中",
        "High": "高",
    },
    "ko": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "사고 또는 추론 (OpenAI 호환만)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* 지원하는 LLM 번역기에만 영향을 줍니다. DeepL / Microsoft / Google에는 적용되지 않습니다.",
        "Off": "끄기",
        "Low": "낮음",
        "Medium": "중간",
        "High": "높음",
    },
    "ru": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "Уровень рассуждений (только для OpenAI-совместимых)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* Влияет только на LLM-переводчики, поддерживающие эту функцию. Не применяется к DeepL / Microsoft / Google.",
        "Off": "Выкл",
        "Low": "Низкий",
        "Medium": "Средний",
        "High": "Высокий",
    },
    "tr": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "Düşünme veya Reasoning (yalnızca OpenAI uyumlu)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* Yalnızca bunu destekleyen LLM çevirmenleri etkiler. DeepL / Microsoft / Google için geçerli değildir.",
        "Off": "Kapalı",
        "Low": "Düşük",
        "Medium": "Orta",
        "High": "Yüksek",
    },
    "zh-CN": {
        "Thinking or Reasoning (OpenAI-compatible only)*": "思考或推理 (仅限 OpenAI 兼容)*",
        "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.": "* 仅影响支持它的 LLM 翻译器。不适用于 DeepL / Microsoft / Google。",
        "Off": "关闭",
        "Low": "低",
        "Medium": "中",
        "High": "高",
    },
}

# The strings to add (in order)
NEW_STRINGS = [
    "Thinking or Reasoning (OpenAI-compatible only)*",
    "* Only affects LLM translators that support it. Does not apply to DeepL / Microsoft / Google.",
    "Off",
    "Low",
    "Medium",
    "High",
]

def add_translations_to_ts(ts_file_path, lang_code):
    """Add new translation entries to a .ts file."""
    with open(ts_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    translations = TRANSLATIONS[lang_code]
    
    # Find the ToolsPage context and the Translator message within it
    # We'll insert after the Translator message block
    translator_pattern = r'(<context>\s*<name>ToolsPage</name>.*?<message>\s*<source>Translator</source>.*?</message>)'
    
    match = re.search(translator_pattern, content, re.DOTALL)
    if not match:
        print(f"WARNING: Could not find ToolsPage/Translator in {ts_file_path}")
        return False
    
    insert_pos = match.end()
    
    # Build new message blocks
    new_blocks = []
    for source in NEW_STRINGS:
        trans = translations[source]
        new_blocks.append(f'''
    <message>
        <source>{source}</source>
        <translation>{trans}</translation>
    </message>''')
    
    new_content = content[:insert_pos] + ''.join(new_blocks) + content[insert_pos:]
    
    with open(ts_file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Added {len(NEW_STRINGS)} translations to {ts_file_path}")
    return True

def main():
    ts_dir = "/home/borrow/comic-translate/resources/translations"
    
    for lang_code in TRANSLATIONS.keys():
        ts_file = os.path.join(ts_dir, f"ct_{lang_code}.ts")
        if os.path.exists(ts_file):
            add_translations_to_ts(ts_file, lang_code)
        else:
            print(f"WARNING: File not found: {ts_file}")

if __name__ == "__main__":
    main()