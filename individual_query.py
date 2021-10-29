import dateutil.parser as parser
import pinyin

def preprocess_input_to_dict(name, nationality=None, gender=None, dob=None):
    def get_year(date):
        try:
            parser_obj = parser.parse(str(date))
            return parser_obj.year
        except:
            return None

    def get_month(date):
        if len(str(date))>4:
            try:
                return parser.parse(str(date)).month
            except:
                return None
        else:
            return None
            
    def get_day(date):
        if len(str(date))>4:
            try:
                return parser.parse(str(date)).day
            except:
                return None
        else:
            return None
    
    def isEnglish(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except:
            return False
        else:
            return True

    if name is not None:
        name_is_english = isEnglish(name)
        if name_is_english is False:
            try:
                name = pinyin.get(name, format='strip', delimiter=' ')
            except:
                name = None
        current_record = {
            'name' : name,
            'year_of_birth': get_year(dob),
            'month_of_birth': get_month(dob),
            'day_of_birth': get_day(dob),
            'gender': gender if gender == 'Male' or gender == 'Female' else None,
            'nationality': nationality
        }
            
    return current_record