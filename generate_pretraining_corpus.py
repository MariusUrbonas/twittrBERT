from pathlib import Path
from multiprocessing import Pool
from argparse import ArgumentParser
from tqdm import tqdm
import bz2
import json
import os
import re

class Tweet_parser:
    @staticmethod
    def get_full_text(tweet_json):
            # check if tweet status is not deleted
            if 'delete' in tweet_json:
                return None
            # check if the detected language for a tweet is english
            if tweet_json['lang'] != 'en':
                return None
            # if its a retweet, check the original tweet for full text
            if 'retweeted_status' in tweet_json:
                tweet_json = tweet_json['retweeted_status']
            if tweet_json['truncated']:
                text = tweet_json['extended_tweet']['full_text']
            else:
                text = tweet_json['text']
            return text

class Tweet_processer:
    @staticmethod
    def replace_url(text):
        url_pattern = '(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]\
                    *[\w@?^=%&/~+#-])?(/([A-Z]|[a-z]|[0-9])*)?'
        return re.sub(url_pattern, '<url>', text)

    @staticmethod
    def replace_nl(text):
        return " ".join(text.split("\n"))

    @staticmethod
    def split_to_sentences(text):
        sentence_pattern = re.compile("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")
        return sentence_pattern.split(text)

    @staticmethod
    def long_enough(sentence, min_length):
        return len(sentence.split()) >= min_length

    @classmethod
    def merge_short_sentences(cls, sentences, min_length):
        # if one of the sentences in the list is too short for current theshold
        # merge it with shortest surrounding sentence
        merged_sentences = []
        i = 0
        last_merged = False
        while i < len(sentences)-1:
            last_merged = False
            if not cls.long_enough(sentences[i], min_length):
                sent = sentences[i] + " " + sentences[i+1]
                merged_sentences.append(sent)
                i += 1
                last_merged = True
            else:
                merged_sentences.append(sentences[i])
            i += 1
        if not last_merged:
            if cls.long_enough(sentences[i], min_length):
                merged_sentences.append(sentences[-1])
            else:
                merged_sentences = merged_sentences[:-1] +\
                                   [merged_sentences[-1]+' '+ sentences[-1]]
        return merged_sentences

    @classmethod
    def preprocess_tweet(cls, tweet_json, min_length, min_num, split):
        text = Tweet_parser.get_full_text(tweet_json)
        if text is None:
            return None
        text = cls.replace_url(text)
        text = cls.replace_nl(text)
        if split:
            sentences = cls.split_to_sentences(text)
        if len(sentences) == min_num:
            return sentences
        if len(sentences) < min_num:
            return None
        sentences = cls.merge_short_sentences(sentences, min_length)
        if len(sentences) < min_num:
            return None
        return sentences


def process_tweets(input): 
    filename, min_sent_length, min_sent_num, split_sent = input
    data = []
    with bz2.open(str(filename), "rt") as bzinput:
        for i, line in enumerate(bzinput):
            tweet_json = json.loads(line)
            tweet_data = Tweet_processer.preprocess_tweet(tweet_json, min_sent_length, min_sent_num, split_sent)
            if tweet_data is not None:
                data.append(tweet_data)
    return data

def stats(data):
    num_tweets = 0
    sent_lenghts = 0
    num_tweets = len(data)
    tweet_lenghts = list(map(len, data))
    return {'num_tweets': num_tweets,
            'mean_lenght': sum(tweet_lenghts)/num_tweets}


def main():
    parser = ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True,
                        help="Top directory containing twitter json.bz2 files")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    parser.add_argument("--min_sent_len", type=int, default=3,
                        help="Minimum word count allowed for a sentence")
    parser.add_argument("--min_sent_num", type=int, default=2,
                        help="Minimum sentence count allowed for a sentence")
    parser.add_argument("--num_tweets", type=int, default=-1,
                        help="Max num tweets required, set to -1 of you want all")
    parser.add_argument("--split_sent", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True) 

    # Get tweet json.bz2 file paths
    fnames = []
    for filename in Path(args.corpus).glob('**/*.json.bz2'):
        fnames.append(filename)

    print(">> Found {} files".format(len(fnames)))

    # Extract relevant text
    data = []
    batch_mult = 4
    if not args.test:
        pbar = tqdm(range(len(fnames)//(args.num_workers*batch_mult)))
        for i in pbar:
            fname_batch = fnames[i:i+(args.num_workers*batch_mult)]
            with Pool(args.num_workers) as p:
                input = list(zip(fname_batch, [args.min_sent_len]*len(fname_batch), [args.min_sent_num]*len(fname_batch), [args.split_sent]*len(fname_batch)))
                output = list(p.imap(process_tweets, input))
                flat_data = [item for sublist in output for item in sublist]    
                data.extend(flat_data)
            pbar.set_description("Data contains %d tweets" % len(data))
            if args.num_tweets != -1 and len(data) > args.num_tweets:
                break

    else:
        print('>> Tesing on 2 files')
        for fname in tqdm(fnames[:2]):
            data.extend(process_tweets((fname,args.min_sent_len,args.min_sent_num)))

    #flat_data = [item for sublist in data for item in sublist]    

    # Write text into file
    st = stats(data)
    num_items = st['num_tweets'] if st['num_tweets'] < args.num_tweets else args.num_tweets
    save_file = str(args.output_dir) + '/corpus_len_{}_min_sent_{}_min_len_{}.txt'.format(num_items, args.min_sent_num, args.min_sent_len)
    with open(save_file, "w") as text_file:
        for tweet in data[:args.num_tweets]:
            for line in tweet:
                text_file.write(line+'\n')
            text_file.write('\n')

    print(st)

if __name__ == '__main__':
    main()