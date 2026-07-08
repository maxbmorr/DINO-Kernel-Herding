import __utils__ as ut

X, Y, paths, classes = ut.create_dataset(
    data_location="training_data",
    bckgnd_rmv = True
)

print("X shape:", X.shape)
print("Y:", Y)
print("Classes:", classes)
print("Paths:", paths)
