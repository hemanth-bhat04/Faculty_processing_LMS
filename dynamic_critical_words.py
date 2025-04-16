import traceback
from concurrent.futures import ThreadPoolExecutor
from difflib import get_close_matches
from functools import lru_cache

import numpy as np
import psycopg2
import requests


class JunkWordProcessor:
    """
    Class for performing cleaning on words.
    """

    junk_keywords = {'subscribe_link', 'real world', 'code try', 'meaning water', 'stated', 'groups', 'helpful', 'uses',
                     'build accurate', 'types', 'models belong', 'allows', 'parts', 'takes', 'wrong statement',
                     'overall idea', '___ mutable', 'techniques comes', 'mentioned', 'none', 'semester', 'videos',
                     'siraj', 'blog post', 'challenging problem', 'youtube channel', 'simpler', 'second trick',
                     'hard work', 'awesome progress', 'nothing', 'formed', 'call', 'find', 'independent',
                     'several ways', 'hyphen', 'occurrences', 'par', 'long list', 'upcoming section', 'corresponds',
                     'common component', 'teach class', 'open mind', 'getting started', 'second challenge', 'thanks',
                     'congratulations', 'general numbers', 'congratulation', 'depth', 'app', 'keep track', 'good luck',
                     'designer', 'heading', 'current value', 'subscribe', 'comment', 'comments', 'email', 'paragraph',
                     'paragraphs', 'full day', 'second command', 'entire file', 'single file', 'subsequent chapters',
                     'large portion', 'current working', 'long file', 'current group', 'default', 'missive',
                     'subscribe button', 'changing cow', 'large number', 'empty line', 'semicolon',
                     'www.it ebooks.info', 'www.it-ebooks.info', 'above-mentioned problems', 'consider', 'used',
                     'achieve', 'vow www.it ebooks.info', 'fully qualified', 'pretty cool', 'technology university',
                     'definitely', 'luck', 'basically', 'really', 'maybe', 'exactly', 'tons', 'perfectly', 'overs',
                     'topic', 'hey', 'world program', 'whole bunch', 'month', 'leeway', 'mistakes'}

    single_word_junk = {'postulate', 'recap', 'aipmt', 'introduction', 'additional', 'cet', 'serve', 'unit',
                        'references', 'choose', 'prepartion', 'background', 'iit-jee', 'problem', 'activities',
                        'solutions', 'match', 'remember', 'marks', 'lets', 'long', 'numerical', 'solutuons',
                        'illustrative', 'experiments', 'blanks', 'important points', 'iit jee', 'overview', 'type',
                        'free', 'based', 'illustration', 'guess', 'miscelleneous', 'features', 'reference', 'solution',
                        'four', 'definitions', 'suppose', 'activity', 'hint', 'definition', 'hence', 'study',
                        'examples', 'e.g', 'project', 'answer', 'imagine', 'postulates', 'explanations', 'table',
                        'observation', 'understanding', 'content', 'rectify', 'proof', 'conclude', 'answers', 'chapter',
                        'encyclopedia', 'eg.', 'figure', 'formula', 'statement', 'lesson', 'contents', 'class', 'given',
                        'sir', 'hitns', 'show', 'chart', 'let us see', 'mark', 'solve', 'summary', 'note',
                        'alternative', 'structure', 'do you know', 'down', 'crossword', 'diagram', 'know this', 'test',
                        'process', 'mv', 'learn', 'suggested', 'fig', 'tick', 'recall', 'axiom', 'principle',
                        'important point', 'source', 'sources', 'remark', 'remarks', 'article', 'axioms', 'fill', 'pu',
                        'correct', 'exercises', 'exercise', 'conclusion', 'step', 'checklistexplanation', 'short',
                        'college', 'consequences', 'studying', 'checklist', 'find out', 'box', 'observe', 'there',
                        'level', 'formulae', 'case', 'fun', 'questions', 'points', 'projects', 'experiment', 'main',
                        'habits', 'hints', 'reasons', 'puc', 'uses ', 'problems', 'let', 'imagination', 'think',
                        'across', 'example', 'chemical properties', 'statements', 'practice', 'habit', 'clues',
                        'wikipedia', 'brief', 'application', 'type', 'types', 'progress', 'appropriate', 'words',
                        'readings', 'websites', 'reading', 'outline', 'review', 'summing', 'model', 'motivation',
                        'preface', 'introductory', 'general', 'bibliography', 'requirement', 'requirements', 'set',
                        'sets', 'proposition', 'program', 'output', 'result', 'page', 'page', 'no.', 'ans.', 'ans',
                        'stage', 'exhibit', 'outcome', 'exercises', 'listing', 'general property', 'general properties',
                        'school', 'distance', 'education', 'some', 'basic', 'facts', 'notes', 'basics', 'self', 'check',
                        'research', 'researches', 'learning', 'objectives', 'objective', 'includes', 'include',
                        'practically', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                        'september', 'october', 'november', 'december', 'less', 'add', 'module', 'downloaded', 'finish',
                        'writing', 'estimated', 'Essay', 'Type', 'short', 'result', 'discussion', 'results',
                        'discussions', 'case study', 'investigation', 'investigations', 'practical', 'practicals',
                        'worksheet', 'worksheets', 'work sheet', 'work sheets', 'name', 'explanation', 'method',
                        'classroom', 'demo', 'demos', 'demonstrations', 'demonstration', 'group', 'groups', 'rule',
                        'works', 'mix', 'mixed', 'mixes', 'caution', 'nature', 'demonstrate', 'key concepts', 'sample',
                        'typical', 'key', 'key terms', 'transform', 'transforming', 'chapter objective',
                        'chapter objectives', 'instruction', 'instructions', 'unit structure', 'tables', 'ask',
                        'basics', 'identify', 'identifying', 'describe', 'evaluate', 'mini-project', 'mini project',
                        'select', 'steps', 'problem definition', 'types', 'hypothesis', 'scope', 'assumption',
                        'assumptions', 'methodology', 'conclusions', 'physics', 'chemistry', 'maths', 'mathematics',
                        'science', 'biology', 'properties', 'glossary', 'direction', 'directions', 'notation',
                        'notations', 'mathematical background', 'syllabus', 'syllabi', 'category', 'categories',
                        'scert telangana', 'concept', 'concepts', 'friend', 'friends', 'scientific', 'physics', 'maths',
                        'chemistry', 'biology', 'mathematics', 'opinion', 'opinions', 'lot', 'lots', 'money',
                        'previous', 'next', 'slide'}

    abusive_words = {
        '"4r5e"', '"5h1t"', '"5hit"', 'a55', 'anal', 'anus', 'ar5e', 'arrse', 'arse', 'arsehole', 'ass', 'assbandit',
        'assbanger', 'assbite', '"ass-fucker"', 'asses', 'assfucker', 'assfukka', 'assfuka', 'asshole', 'asshol',
        'assholes', 'asswhole', 'a_s_s', 'assclown ', 'asscock ', 'asscracker', 'assfuck', 'assgoblin', 'asshat',
        'asshead', 'asshopper', 'assjacker', 'asslick', 'asslicker', 'assmonkey', 'assmunch', 'assmuncher', 'assnigger',
        'asspirate', 'assshit', 'asssucker', 'asswipe', 'asswiper', '"b!tch"', 'b00bs', 'b17ch', 'b1tch', 'ballbag',
        'ballsack', 'bastard', 'beastial', 'beastiality', 'bellend', 'bestial', 'bestiality', '"bi+ch"', 'biatch',
        'bitch', 'bitcher', 'bitchers', 'bitches', 'bitchin', 'bitchn', 'bitching', '"bloody hell"', '"blow job"',
        'blowjob', 'blowjobs', 'boiolas', 'bollock', 'bollok', 'boner', 'boob', 'boobs', 'booobs', 'boooobs',
        'booooobs', 'booooooobs', 'breasts', 'breast', 'buceta', 'bugger', 'bum', '"bunny fucker"', 'butt', 'butthole',
        'buttmuch', 'buttmunch', 'buttplug', 'c0ck', 'c0cksucker', '"carpet muncher"', 'cawk', 'chink', 'cipa', 'cl1t',
        'clit', 'clitoris', 'clits', 'cnut', 'cock', '"cock-sucker"', 'cockface', 'cockhead', 'cockmunch',
        'cockmuncher', 'cocks', 'cocksuck', 'cocksucked', 'cocksucker', 'cocksucking', 'cocksucks', 'cocksuka',
        'cocksukka', 'cok', 'cokmuncher', 'coksucka', 'coon', 'cox', 'crap', 'cum', 'cummer', 'cumming', 'cums',
        'cumshot', 'cunilingus', 'cunillingus', 'cunnilingus', 'cunt', 'cuntlick', 'cuntlicker', 'cuntlicking', 'cunts',
        'cyalis', 'cyberfuc', 'cyberfuck', 'cyberfucked', 'cyberfucker', 'cyberfuckers', 'cyberfucking', 'd1ck', 'dick',
        'dicksuck', 'dicksucking', 'dicksuk', 'diksuk', 'dickhead', 'dildo', 'dildolick', 'dildosuck', 'dildos', 'dink',
        'dinks', 'dirsa', 'dlck', 'dickjuice', 'dickmilk', '"dog-fucker"', 'doggin', 'dogging', 'doggystyle',
        'doggyshtyle', 'donkeyribber', 'doosh', 'duche', 'dyke', '"dry hump"', 'ejaculate', 'ejaculated', 'ejaculates',
        'ejaculating', 'ejaculatings', 'ejaculation', 'ejakulate', '"f u c k"', '"f u c k e r"', 'f4nny', 'fag',
        'fagging', 'faggitt', 'faggot', 'faggs', 'fagot', 'fagots', 'fags', 'fanny', 'fannyflaps', 'fannyfucker',
        'fanyy', 'fatass', 'fcuk', 'fcuker', 'fcuking', 'feck', 'fecker', 'felching', 'fellate', 'fellatio',
        'fingerfuck', 'fingerfucked', 'fingerfucker', 'fingerfuckers', 'fingerfucking', 'fingerfucks', 'fistfuck',
        'fistfucked', 'fistfucker', 'fistfuckers', 'fistfucking', 'fistfuckings', 'fistfucks', 'flange', 'fook',
        'fooker', 'fuck', 'fucka', 'fucked', 'fucker', 'fuckers', 'fuckhead', 'fuckheads', 'fuckin', 'fuckn', 'fucking',
        'fuckings', 'fuckingshitmotherfucker', 'fuckme', 'fucks', 'fuckwhit', 'fuckwit', '"fudge packer"',
        'fudgepacker', 'fuk', 'fuker', 'fukker', 'fukkin', 'fuks', 'fukwhit', 'fukwit', 'fux', 'fux0r', 'f_u_c_k',
        'gangbang', '"gangbanged "', '"gangbangs "', 'gaylord', 'gaysex', 'goatse', '"god-dam"', '"god-damned"',
        'goddamn', 'goddamned', '"hard core sex"', 'hardcoresex', 'hell', 'heshe', 'hoar', 'hoare', 'hoer', 'homo',
        'hore', 'horniest', 'horny', 'hotsex', '"jack-off "', 'jackoff', 'jap', '"jerk-off "', 'jism', '"jiz "',
        '"jizm "', 'jizz', 'Juicyass', 'Juicypuss', 'juicypussy', 'juicepus', 'juicepuss', 'juicepussy', 'juiceball',
        'juicyball', 'juicyballs', 'kawk', 'knobead', 'knobed', 'knobend', 'knobhead', 'knobjocky', 'knobjokey', 'kock',
        'kondum', 'kondums', 'kum', 'kummer', 'kumming', 'kums', 'kunilingus', '"l3i+ch"', 'l3itch', 'labia', 'lmfao',
        'lust', 'lusty', 'lusti', 'lusting', 'm0f0', 'm0fo', 'm45terbate', 'ma5terb8', 'ma5terbate', 'masturbat',
        'ma5turbate', 'm45turbate', 'm45turb8', 'ma5turb8', 'masochist', '"master-bate"', 'masterb8', '"masterbat*"',
        'masterbat3', 'masterbate', 'masterbation', 'masterbations', 'masturbate', '"mo-fo"', 'mof0', 'mofo',
        'mothafuck', 'mothafucka', 'mothafuckas', 'mothafuckaz', 'mothafucked', 'mothafucker', 'mothafuckers',
        'mothafuckin', 'mothafucking', 'mothafuckings', 'mothafucks', '"mother fucker"', 'motherfuck', 'motherfucked',
        'motherfucker', 'motherfuckers', 'motherfuckin', 'motherfucking', 'motherfuckings', 'motherfuckka',
        'motherfucks', 'muff', 'mutha', 'muthafecker', 'muthafuckker', 'muther', 'mutherfucker', 'n1gga', 'n1gger',
        'nazi', 'nigg3r', 'nigg4h', 'nigga', 'niggah', 'niggas', 'niggaz', 'nigger', 'niggers', 'nob', '"nob jokey"',
        'nobhead', 'nobjocky', 'nobjokey', 'numbnuts', 'nutsack', 'nipple', 'nipples', 'orgasim', 'orgasims', 'orgasm',
        'orgasms', 'panties', 'playboy', 'p0rn', 'porno', 'p0rn0', 'pawn', 'pecker', 'penis', 'penisfucker', 'phonesex',
        'phuck', 'phuk', 'phuked', 'phuking', 'phukked', 'phukking', 'phuks', 'phuq', 'pigfucker', 'pimpis', 'piss',
        'pissed', 'pisser', 'pissers', 'pisses', 'pissflaps', 'pissin', 'pissing', 'pissoff', 'porn', 'porno',
        'pornography', 'pornos', 'pron', 'pube', 'pusse', 'pussi', 'pussies', 'pussy', 'pussys', 'retard', 'rimjaw',
        'rimming', 'rimjob', '"s hit"', '"s.o.b."', 'schlong', 'screwing', 'scroat', 'scrote', 'scrotum', 'semen',
        'sex', 'sexy', 'sexsex', 'nicesex', 'rodsex', 'sexophiliac', 'sexbastard', '"sh!+"', '"sh!t"', 'sh1t', 'shag',
        'shagger', 'shaggin', 'shagging', 'shemale', '"shi+"', 'shit', 'shitdick', 'shite', 'shited', 'shitey',
        'shitfuck', 'shitfull', 'shithead', 'shiting', 'shitings', 'shits', 'shitted', 'shitter', 'shitters',
        'shitting', 'shittings', 'shitty', 'sisterfucker', 'sistafucker', 'sistafuker', 'sisterfuker', 'sisterfukker',
        'sistafukker', 'skank', 'slut', 'sluts', 'smegma', 'smut', '"son-of-a-bitch"', 'sonofabitch', 'spac', 'spank',
        's_h_i_t', 'suck', 'sucker', 'suckpenis', 'suckdildo', 'suckdick', 'suckdik', 'sucktit', 'sucktitti',
        'sucktitt', 'sucktitts', 'sucktitty', 'suckboob', 'sucknipple', 'sodomy', 'sodomize', 't1tt1e5', 't1tties',
        'teets', 'teez', 'testical', 'testicle', 'tit', 'titfuck', 'tits', 'titt', 'titty', 'tittys', 'tittie5',
        'tittiefucker', 'titties', 'tittyfuck', 'tittywank', 'titwank', 'testicle', 'tosser', 'turd', 'tw4t', 'twat',
        'twathead', 'twatty', 'twunt', 'twunter', 'v14gra', 'v1gra', 'vagina', 'viagra', 'vulva', 'w00se', 'wang',
        'wank', 'wanker', 'wanky', 'whoar', 'whore', 'willies', 'willy', 'xrated', 'xxx', 'bollox', 'clusterfuck',
        'dipshit', '"behen chod"', '"mader chod"', '"behen chot"', '"mader chot"', 'behenchod', 'madarchod',
        'behenchot', 'madarchot', 'behnchod', 'behnchot', 'madrchod', 'madrchot', 'maadrchod', 'maadrchot', 'maderchot',
        'maderchod', 'haramzade', 'haramsade', 'haramkor', 'haraami', '"haraam zade"', '"haraam zaada"', 'chutia',
        'chutiya', 'mamme', 'kamina', 'kamine', 'kaminey', 'choot', 'gaand', 'gaandu', 'bhaichod', 'bhaichot',
        '"bhai chod"', '"bhai chot"'}

    extended_stop_words = {'called', 'happened', 'go', 'see', 'will', 'false', 'true', 'known', 'kept', 'keep', 'units',
                           'unit', 'way', 'ways', 'per', 'ok', 'versus', 'vs', 'stuff', 'good', 'next', 'previous',
                           'item', 'side', 'listing', 'comma', 'colon', 'particular', 'a', 'about', 'above',
                           'according', 'across', 'after', 'again', 'against', 'all', 'almost', 'along', 'already',
                           'also', 'although', 'always', 'am', 'amen', 'among', 'amongst', 'an', 'between', 'and',
                           'another', 'any', 'anybody', 'anyone', 'anything', 'anyways', 'are', "aren't", 'arent',
                           'around', 'as', 'aside', 'asked', 'asking', 'at', 'affecting', 'atop', 'available', 'avoid',
                           'avoids', 'be', 'because', 'become', 'becomes', 'been', 'before', 'being', 'below', 'both',
                           'buildup', 'but', 'by', 'came', 'can', "can't", 'cannot', 'cant', 'caught', 'consistent',
                           'could', "couldn't", 'couldnt', 'custom', 'darn', 'did', "didn't", 'didnt', 'do', 'does',
                           "doesn't", 'doesnt', 'doing', "don't", 'dont', 'down', 'during', 'each', 'either', 'else',
                           'establish', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'few', 'fig',
                           'fine', 'finer', 'following', 'for', 'from', 'further', 'furthering', 'furthers', 'get',
                           'gets', 'goes', 'going', 'gone', 'gonna', 'gosh', 'had', "hadn't", 'hadnt', 'has', "hasn't",
                           'hasnt', 'have', "haven't", 'havent', 'having', 'he', "he'd", "he'll", "he's", 'hearty',
                           'hed', 'hell', 'her', 'here', "here's", 'herein', 'heres', 'hers', 'herself', 'hes', 'him',
                           'himself', 'his', 'how', "how's", 'hows', 'howz', 'i', "i'd", "i'll", "i'm", "i've", 'id',
                           'if', 'ill', 'im', 'immediately', 'in', 'inner', 'into', 'is', "isn't", 'isnt', 'it', "it's",
                           'its', 'itself', 'ive', 'know', 'let', "let's", 'lets', 'made', 'me', 'merely', 'more',
                           'most', 'much', 'must', "mustn't", 'mustnt', 'my', 'myself', 'never', 'new', 'newer', 'no',
                           'nor', 'not', 'now', 'nowadays', 'of', 'off', 'okay', 'on', 'once', 'ones', 'only', 'or',
                           'originally', 'other', 'otherwise', 'ought', 'our', 'ours', 'ourself', 'ourselves', 'out',
                           'over', 'own', 'plenty', 'pooh', 'quite', 'rarely', 'said', 'same', 'says', 'seem', 'seems',
                           "shan't", 'shant', 'she', "she'd", "she'll", "she's", 'shed', 'shes', 'should', "shouldn't",
                           'shouldnt', 'shut', 'so', 'some', 'somebody', 'someday', 'someone', 'something', 'sometime',
                           'sometimes', 'somewhere', 'soon', 'sorry', 'still', 'straightforward', 'successful', 'such',
                           'tact', 'take', 'tactful', 'than', 'that', "that's", 'thats', 'the', 'their', 'theirs',
                           'them', 'themselves', 'then', 'there', "there's", "there've", 'thereabout', 'thereafter',
                           'therefore', 'theres', 'thereve', 'these', 'they', "they'd", "they'll", "they're", "they've",
                           'theyd', 'theyll', 'theyre', 'theyve', 'thing', 'things', 'this', 'those', 'thou', 'through',
                           'thus', 'to', 'today', 'tomorrow', 'too', 'unbeknownst', 'under', 'until', 'up', 'us',
                           'utmost', 'utter', 'very', 'want', 'wanted', 'was', "wasn't", 'wasnt', 'wasntw', 'we',
                           "we'd", "we'll", "we're", "we've", 'wed', 'weekly', 'weird', 'well', 'went', 'were',
                           "weren't", 'werent', 'weve', 'what', "what's", 'whatever', 'whats', 'whatsoever', 'when',
                           "when's", 'whens', 'where', "where's", 'wherefrom', 'wherein', 'wheres', 'wherever',
                           'whether', 'which', 'whichever', 'while', 'whilst', 'who', "who's", 'whoa', 'whom', 'whos',
                           'whose', 'why', "why's", 'whys', 'with', "won't", 'wont', 'would', "would've", "wouldn't",
                           'wouldnt', 'wouldve', 'yeah', 'yesterday', 'you', "you'd", "you'll", "you're", "you've",
                           'youd', 'youll', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'youve', 'however',
                           'later', 'unfortunately', 'consequently', 'chapter', 'got', 'theyve', 'name', 'introduction',
                           'many', 'different', 'various', 'suitable', 'one', 'two', 'three', 'four', 'five', 'six',
                           'seven', 'eight', 'nine', 'ten', 'zero', 'important', 'importance', 'part', 'say', 'modern',
                           'definition', 'possible', 'factor', 'factors', 'both', 'first', 'apart', 'great', 'major',
                           'minor', 'using', 'last', 'nearly', 'main', 'exhibit', 'exhibits', 'figure', 'figures',
                           'table', 'tables', 'useful', 'use', 'example', 'examples', 'specific', 'special', 'may',
                           'include', 'includes', 'including', 'given', 'look', 'just', 'tell', 'reached', 'others',
                           'way', 'huge', 'basic', 'safe', 'heavy', 'little', 'certain', 'certainly', 'piece',
                           'significant', 'separate', 'nahi', 'guy', 'guys'}

    all_junk = junk_keywords | extended_stop_words | abusive_words | single_word_junk

    @staticmethod
    def check_section_topics(phrase: str) -> bool:
        """
        Function to eliminate junk phrases.
        Checks if a phrase can be found in a text_book.
        Returns the phrase if it is correct and empty string if it is junk.
        """

        # Text Book sections stored in section_topics core in Solr
        sec_topics_url = 'http://164.52.201.193:8983/solr/' + 'section_topics'

        try:
            query_params = {
                'q': '"' + phrase.strip().lower() + '"',
                'qf': 'section',
                'defType': 'edismax',
                'indent': 'on',
                'fl': 'id',
                'wt': 'json',
                'rows': 1,
            }
            r = requests.get('%s/select' % sec_topics_url, params=query_params, timeout=10)
            r_docs = eval(r.text).get('response', {}).get('docs', {})

            if len(r_docs) > 0:
                return True
            # elif not is_recheck:
            #     return check_section_topics(phrase,is_recheck=True)
            else:
                return False

        except requests.exceptions.HTTPError:
            raise Exception("Http Error")
        except requests.exceptions.ConnectionError:
            raise Exception("Error Connecting")
        except requests.exceptions.Timeout:
            raise Exception("Timeout Error")

    @staticmethod
    @lru_cache(maxsize=None)
    def valid_word_check(input_word: str) -> str:
        """
        Function checks if the word is valid by checking text book as well all junk phrases.
        """

        input_word = input_word.lower().strip()
        if not JunkWordProcessor.check_section_topics(input_word) or input_word in JunkWordProcessor.all_junk:
            return ''
        else:
            return input_word

    @staticmethod
    def clean_words(input_words: list) -> list:

        with ThreadPoolExecutor() as executor1:
            result = executor1.map(JunkWordProcessor.valid_word_check, input_words)

        return [x for x in result if x.strip()]


class DataBase:
    """
    Database class to perform DB operations.
    """

    def __init__(self):
        self._host = "164.52.194.25"
        self._dbname = "piruby_automation"
        self._password = "piruby@157"
        self._port = "5432"
        self._user = "postgres"
        self.__conn = psycopg2.connect(dbname=self._dbname, user=self._user,
                                       host=self._host, password=self._password, port=self._port)

    # def get_video_ids(self, five_min_table: str = "private_videomaster_m", limit: int = None) -> list:
    def get_video_ids(self, five_min_table: str = "cs_new_5m_temp_new", limit: int = None) -> list:

        cursor = self.__conn.cursor()
        if limit is None:
            query = "SELECT DISTINCT video_id FROM " + five_min_table + ";"
        else:
            query = "SELECT DISTINCT video_id FROM " + five_min_table + " LIMIT " + str(limit) + ";"
        cursor.execute(query)
        vids = cursor.fetchall()
        cursor.close()

        return list(set(x[0] for x in vids))

    # def get_5min_keywords(self, video_id: str, five_min_table: str = "private_videomaster_m") -> list:
    def get_5min_keywords(self, video_id: str, five_min_table: str = "cs_new_5m_temp_new") -> list:

        if not video_id.strip():
            return []

        cursor = self.__conn.cursor()

        query = "SELECT critical_keywords from " + five_min_table + \
                " WHERE video_id='%s' and critical_keywords is not null ORDER BY _offset" % video_id
        cursor.execute(query)
        chunk_keywords = cursor.fetchall()
        cursor.close()

        return [x[0] for x in chunk_keywords]

    # Critical All Keywords
    # def get_critical_all_keywords(self, five_min_table: str = "private_videomaster_m", limit: int = None) -> list:
    def get_critical_all_keywords(self, five_min_table: str = "cs_new_5m_temp_new", limit: int = None) -> list:

        if limit is None:
            query = "SELECT DISTINCT video_id,critical_all_keywords from " + five_min_table + ";"
        else:
            query = "SELECT DISTINCT video_id,critical_all_keywords from " + five_min_table + \
                    " LIMIT " + str(limit) + ";"

        cursor = self.__conn.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()
        return records

    # def update_critical_all_keywords(self, video_id: str, critical_all_keywords: list,
    #                                  five_min_table: str = "private_videomaster_m") -> None:
    def update_critical_all_keywords(self, video_id: str, critical_all_keywords: list,
                                      five_min_table: str = "cs_new_5m_temp_new") -> None:

        cursor = self.__conn.cursor()
        query = "UPDATE " + five_min_table + " SET critical_all_keywords = %s, critical_all_keywords_count = %s" \
                                             " WHERE video_id = %s"
        cursor.execute(query, (critical_all_keywords, len(critical_all_keywords), video_id))
        self.__conn.commit()
        cursor.close()

    def close(self):
        self.__conn.close()

    def __str__(self):
        return "{} : IP: {}, User: {}, DB: {}".format(self.__class__.__name__, self._host, self._user, self._dbname)


def process_keywords(chunked_words_list: list) -> list:

    final_weighted_words = {}
    for word_list in chunked_words_list:
        # print("Keywords in 5 Min Chunk: {}".format(len(word_list)))
        cleaned_word_list = JunkWordProcessor.clean_words(word_list)
        # print(JunkWordProcessor.valid_word_check.cache_info())
        weighted_words = get_dynamic_weights(cleaned_word_list)
        for word_weight_tuple in weighted_words:
            word, weight = word_weight_tuple
            similar_words = get_close_matches(word, final_weighted_words.keys(), cutoff=0.8)
            if len(similar_words) > 0:
                final_weighted_words[similar_words[0]] = final_weighted_words[similar_words[0]] + weight
            else:
                final_weighted_words[word] = weight

    final_weighted_words = sorted(final_weighted_words.items(), key=lambda x: x[1], reverse=True)

    if len(chunked_words_list) > 5:
        cutoff_thresh = 2.001
    elif len(chunked_words_list) > 1:
        cutoff_thresh = 1.85
    else:
        cutoff_thresh = 1.6
    final_keywords = [x[0] for x in final_weighted_words if x[1] >= cutoff_thresh]

    # print(final_keywords)
    print("Minutes: {}, Total Keywords: {}, Filtered Keywords: {}".format(
        (len(chunked_words_list)*5), len(final_weighted_words), len(final_keywords)))

    return final_keywords


def get_dynamic_weights(words: list) -> list:
    """
    Function takes in a list of keywords that are already reverse sorted in terms of importance,
    i.e, Important words first.

    Then weights the keywords dynamically between 1 - 2

    Returns a list of (word, weight) tuple pairs.
    """
    return list(zip(words, reversed(list(np.linspace(1, 2, len(words))))))


def get_lines_from_file_as_list(text_filename: str) -> list:
    """
    Function to read lines from a txt file and return them as a list
    """

    with open(str(text_filename), 'r') as f:
        lines = f.readlines()

    return [line.replace('\n', '') for line in lines]


def main():
    """
    Main function to process videos and formulate critical all keywords from 5min keywords.
    """

    db = DataBase()
    print(db)
    videos = db.get_video_ids()

    # LOGGING_FILE = 'completed_dynamic_keyword_formation_vidsMay13.txt'
    # LOGGING_FILE = 'completed_dynamic_keyword_formation_vidsDec24.txt'
    LOGGING_FILE = 'completed_dynamic_keyword_formation_vidsJan25.txt'

    # COMPLETED_VIDEOS = set(get_lines_from_file_as_list(LOGGING_FILE))
    COMPLETED_VIDEOS = set()

    COUNT = 0

    def process_video(video_id: str) -> None:

        nonlocal COUNT
        COUNT += 1
        print("Completed: {}".format(COUNT))

        if video_id in COMPLETED_VIDEOS:
            return

        with open(LOGGING_FILE, 'a') as openfile:
            openfile.write(video_id)
            openfile.write('\n')

        try:
            chunked_keywords_list = db.get_5min_keywords(video_id=video_id)
            # print("5 Minute Chunks: {}".format(len(chunked_keywords)))
            updated_all_keywords = process_keywords(chunked_keywords_list)
            if not updated_all_keywords:
                return
            db.update_critical_all_keywords(video_id, updated_all_keywords)
        except Exception as exc:
            print("Video: {}".format(video_id))
            print(traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__))

    with ThreadPoolExecutor() as executor2:
        executor2.map(process_video, videos)

    db.close()


if __name__ == '__main__':
    main()
