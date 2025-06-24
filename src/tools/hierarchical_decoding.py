import pandas as pd
from typing import List, Dict, Any, Optional

class PTBXLSCPHierarchy:
    def __init__(self, scp_statements_path: str):
        self.df = pd.read_csv(scp_statements_path, index_col='scp_code')
        # Fill missing values with empty strings for convenience
        self.df = self.df.fillna('')
        # Dictionaries for fast lookup
        self.code2subclass = self.df['diagnostic_subclass'].to_dict()
        self.code2class = self.df['diagnostic_class'].to_dict()
        self.subclass2class = self.df.set_index('diagnostic_subclass')['diagnostic_class'].to_dict()

    def decode(self, pred_codes: List[str]) -> Dict[str, Dict[str, List[str]]]: #Forming the hierarchy: diagnostic_class -> diagnostic_subclass -> [scp_code]
        hierarchy = {}
        for code in pred_codes:
            subclass = self.code2subclass.get(code, '')
            dclass = self.code2class.get(code, '')
            if not dclass:
                dclass = 'Other'
            if dclass not in hierarchy:
                hierarchy[dclass] = {}
            if subclass not in hierarchy[dclass]:
                hierarchy[dclass][subclass] = []
            hierarchy[dclass][subclass].append(code)
        return hierarchy

    def pretty_print(self, hierarchy: Dict[str, Dict[str, List[str]]], code2name: Optional[Dict[str,str]] = None): #Pretty print of the hierarchy for the user
        for dclass, subclasses in hierarchy.items():
            print(f'{dclass}')
            for subclass, codes in subclasses.items():
                if subclass:
                    print(f'  └─ {subclass}')
                for code in codes:
                    name = code2name[code] if code2name and code in code2name else code
                    if subclass:
                        print(f'      └─ {name}')
                    else:
                        print(f'  └─ {name}')

    def get_multilabels(self, pred_codes: List[str]) -> Dict[str, List[str]]: #Getting the unique diagnostic_class and diagnostic_subclass for multilabel analysis
        classes = set()
        subclasses = set()
        for code in pred_codes:
            subclasses.add(self.code2subclass.get(code, ''))
            classes.add(self.code2class.get(code, ''))
        return {
            'diagnostic_class': sorted([c for c in classes if c]),
            'diagnostic_subclass': sorted([s for s in subclasses if s])
        }

# Example usage
if __name__ == '__main__':
    # Paths to files
    scp_statements_path = 'data/raw/ptb-xl/scp_statements.csv'
    # List of predicted scp_code (e.g. after inference)
    pred_codes = ['IMI', 'AMI', 'AFIB']
    # (optional) dictionary scp_code -> name
    code2name = {'IMI': 'Inferior MI', 'AMI': 'Anterior MI', 'AFIB': 'Atrial Fibrillation'}
    hier = PTBXLSCPHierarchy(scp_statements_path)
    hierarchy = hier.decode(pred_codes)
    hier.pretty_print(hierarchy, code2name=code2name)
    multilabels = hier.get_multilabels(pred_codes)
    print('diagnostic_class multilabels:', multilabels['diagnostic_class'])
    print('diagnostic_subclass multilabels:', multilabels['diagnostic_subclass']) 