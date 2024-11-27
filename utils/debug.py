from json import dumps

def tolist(x):
    return x.detach().cpu().numpy().tolist()

class debug_class:
    def __init__(self,*args, mode = 'plotly', type = None):

        if type == None:
            self.x = {
                "kind": {mode: True},
                "data": [{'y':tolist(arg)} for arg in args]
            }

        elif type == 'heatmap':
            self.x = {
                "kind": {
                    mode: True
                },
                
                "data": [
                    {
                        "type": 'heatmap',
                        "x": [x for y in list(range(len(arg))) for x in list(range(len(arg[0])))],
                        "y": [y for y in list(range(len(arg))) for x in list(range(len(arg[0])))],
                        "z": tolist(arg.reshape(-1))
                    }
                    for arg in args
                ]
            }

        elif type == 'mesh3d':
            self.x = {
                "kind": {
                    mode: True
                },
                
                "data": [
                    {
                        "type": 'mesh3d',
                        "x": [x for y in list(range(len(arg))) for x in list(range(len(arg[0])))],
                        "y": [y for y in list(range(len(arg))) for x in list(range(len(arg[0])))],
                        "z": tolist(arg.reshape(-1))
                    }
                    for arg in args
                ]
            }

        elif type == 'table':
            self.x = {
                "kind":{
                    "table": True
                },
                "rows": [
                   {m:elm for m,elm in enumerate(tolist(row))} for arg in args for row in arg 
                ]
            }
    def __call__(self):
        return dumps(self.x)
