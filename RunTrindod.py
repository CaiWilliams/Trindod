from Trindod import *
import os
import ujson


class LCOERun:

    def __init__(self, filename, paneldatafile):
        self.filename = filename
        self.panel_datafile = paneldatafile

    def mod_pop(self, key, value):
        exp = LCOE(self.filename, self.panel_datafile)
        with open(self.filename + '.json') as params:
            params_dict = ujson.load(params)
            params_dict[key][0] = value
            with open(self.filename + 'TEMP' + '.json', 'w') as outfile:
                ujson.dump(params_dict, outfile)

        filename_temp = self.filename + 'TEMP'
        exp.Q = Que(filename_temp, self.panel_datafile)
        pop_key = list(np.where(np.array(exp.Q.key) == str(key)))[0][0]
        exp.Q.value[pop_key] = np.arange(1, value + 1, 1)
        exp.Q.filename = self.filename
        exp.Q.gen_file()
        exp.Q.save_que()
        os.remove(filename_temp + '.json')
        return

    def run(self):
        exp = LCOE(self.filename, self.panel_datafile)
        exp.generate_jbs()
        exp.load_jbs()
        x = pd.DataFrame(exp.Q.Jobs[0])
        x = x.iloc[0]
        x.to_json('job.json')
        #exp.run()
        return

    def re_run(self):
        exp = LCOE(self.filename, self.panel_datafile)
        exp.load_jbs()
        exp.run()
        return


if __name__ == '__main__':
    #experiment = LCOERun('Experiments/TwoPanTypTwoPrjLoc/TwoPanTypTwoPrjLoc','Data/PanelData.csv')
    experiment = LCOERun('Experiments/TwoPanTypTwoPrjLoc/TwoPanTypTwoPrjLoc', 'Data/PanelData.csv')
    experiment.run()
