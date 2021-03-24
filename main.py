mappings = {'🗓':'anticipation', '🤞':'anticipation', '😡':'colère', '🤬':'colère', '👿':'colère', '😾':'colère',
           '🖕':'colère', '😤':'colère', '😠':'colère', '💪':'confiance', '🤝':'confiance', '👍':'confiance',
           '👌':'confiance', '😒':'déception', '👎':'déception', '😞':'déception', '🙄':'déception', '😖':'déception',
           '😕':'déception', '🤮':'déception', '🤢':'déception', '😘':'joie', '❤️':'joie', '💋':'joie', '💓':'joie',
            '👍':'joie', '👌':'joie', '😻':'joie', '💏':'joie', '💖':'joie', '🤩':'joie', '💕':'joie', '🤗':'joie',
            '💑':'joie', '🎊':'joie', '🎉':'joie', '😍':'joie', '💗':'joie', '👏':'joie', '🌞':'joie', '😨':'peur',
           '😮':'surprise', '😲':'surprise', ':(':'tristesse', '🙁':'tristesse', '😢':'tristesse', ':-(':'tristesse',
           '😞':'tristesse', '💔':'tristesse', '😿':'tristesse'}


def extract_content_emotions_with_emojis(file_name):
    for row in open(file_name, "r",encoding='utf-16'):
        yield (row.split('\t')[0],
             [(s, mappings[s]) for s in row.split('\t')[0] if s in mappings])
        
        
def preprocessing_with_emojis(file_name):
    content_emotions = []
    for row in extract_content_emotions_with_emojis(file_name):
        content_emotions.append(row)
    content_emotions = [x for x in content_emotions if simple_preprocess(x[0], deacc=True)!=[] and x[1]!=[]]
    emotions = [list(set(x[1])) for x in content_emotions]
    content = [x[0] for x in content_emotions]
    return content, emotions
  
  
if __name__ == '__main__':
