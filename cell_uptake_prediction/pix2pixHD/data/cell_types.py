"""
Cell type mapping 

"""


CELL_TYPES = [
    'SKBR3', '3LL', '4T1','Hct116', 'MCF','CT26','A375', 'PC3',
    'MB49', 'A549', 'SW1990' 'B16', 'Hep3B', 'Hela',
    '231', 'HepG2','K180', 'Pan02','SCLC', 'MC38',    
]

CELL_TYPE_TO_IDX = {name: i + 1 for i, name in enumerate(CELL_TYPES)}  
IDX_TO_CELL_TYPE = {i + 1: name for i, name in enumerate(CELL_TYPES)}
NUM_CELL_TYPES = len(CELL_TYPES)  