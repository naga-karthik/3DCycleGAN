import sys
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.filereader import read_dicomdir

pth_name = "/home/karthik/Desktop/For_Naga_Enamundram/PA_ANON/PA000000/"
folder_name = "AX_T1_6MM/"
file_name = "IM000000.dcm"
file = pth_name+folder_name+file_name

filename = get_testdata_file("rtplan.dcm")
ds = dcmread(filename)
print(ds.PatientName)

# filepath = get_testdata_file(pth_name+folder_name)
# dicom_dir = read_dicomdir(filepath)
sys.exit()

ds = dcmread(path)
ds_arr = ds.pixel_array     # converting it into numpy

plt.figure()
plt.imshow(ds_arr, cmap='gray')
plt.show()