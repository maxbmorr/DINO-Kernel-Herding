import DINO as DN
import __utils__ as ut

def main(data_location, bckgnd_rmv = True):
    if bckgnd_rmv == True:
        no_bg_location = "training_data_no_bg"
        ut.rmv_bckgnd(data_location, no_bg_location)
        data_location = no_bg_location
    X, Y, paths, data = DN.DINO_Vector(data_location)
    return X, Y, paths, data

print(main(data_location = "training_data", bckgnd_rmv = True))
