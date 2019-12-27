# coding=utf-8

"""
Pytorch with BERT

Returns:
    [None] -- [Just a practice]
"""
import torch
from IPython.display import clear_output
from transformers import BertTokenizer


def BERT_tokenizer(PRETRAINED_MODEL_NAME):
    # 取得此預訓練模型所使用的 tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    clear_output()
    print("PyTorch 版本：", torch.__version__)
    vocab = tokenizer.vocab
    print("字典大小：{}\n".format(len(vocab)))
    print("-"*80)

    return tokenizer

def sentence_segmentation(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    indexs = tokenizer.convert_tokens_to_ids(tokens)

    print(sentence)
    print(tokens[:10], '...')
    print(indexs[:10], '...')
    print("-"*80)

    return indexs, tokens

def predict_masked_language_model(PRETRAINED_MODEL_NAME, indexs, tokens):
    from transformers import BertForMaskedLM

    # 除了 tokens 以外我們還需要辨別句子的 segment ids
    tokens_tensor = torch.tensor([indexs])  # (1, seq_len)
    segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
    maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
    clear_output()

    # 使用 masked LM 估計 [MASK] 位置所代表的實際 token 
    maskedLM_model.eval()
    with torch.no_grad():
        outputs = maskedLM_model(tokens_tensor, segments_tensors)
        predictions = outputs[0]
        # (1, seq_len, num_hidden_units)
    del maskedLM_model

    # 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
    masked_index = 5
    k = 3
    probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
    predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

    # 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
    print("輸入 tokens ：", tokens[:10], '...')
    print('-' * 80)
    for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
        tokens[masked_index] = t
        print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokens[:10]), '...')

def install_bertviz():
    # 安裝 BertViz
    import sys
    if not 'bertviz_repo' in sys.path:
        sys.path += ['bertviz_repo']

    # import packages
    from bertviz.pytorch_transformers_attn import BertModel, BertTokenizer
    from bertviz.head_view import show

    # 在 jupyter notebook 裡頭顯示 visualzation 的 helper
    def call_html():
        import IPython
        display(IPython.core.display.HTML('''
                <script src="/static/components/requirejs/require.js"></script>
                <script>
                requirejs.config({
                    paths: {
                    base: '/static/base',
                    "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
                    jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
                    },
                });
                </script>
                '''))

    clear_output()

def bert_visualization(PRETRAINED_MODEL_NAME):
    # 記得我們是使用中文 BERT
    model_type = 'bert'

    bertviz_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
    bertviz_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # 情境 1 的句子
    sentence_a = "胖虎叫大雄去買漫畫，"
    sentence_b = "回來慢了就打他。"
    call_html()
    show(bertviz_model, model_type, bertviz_tokenizer, sentence_a, sentence_b)
    # 情境 2 的句子
    sentence_a = "妹妹說胖虎是「胖子」，"
    sentence_b = "他聽了很不開心。"

    call_html()
    show(bertviz_model, model_type, bertviz_tokenizer, sentence_a, sentence_b)

if __name__=="__main__":

    # assign BERT-BASE pre-trained model
    PRETRAINED_MODEL_NAME = "bert-base-chinese"

    # get tokenizer
    tokenizer = BERT_tokenizer(PRETRAINED_MODEL_NAME)

    # get segmented words with indexs from sentence
    sentence = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
    indexs, tokens = sentence_segmentation(sentence, tokenizer)

    """
    這段程式碼載入已經訓練好的 masked 語言模型並對有 [MASK] 的句子做預測
    """
    predict_masked_language_model(PRETRAINED_MODEL_NAME, indexs, tokens)

    # install BertViz
    install_bertviz()

    # Bert attention visualization
    bert_visualization(PRETRAINED_MODEL_NAME, )

