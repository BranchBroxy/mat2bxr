import numpy as np
import h5py  # hdf5
import scipy.io                                                            

def import_mat(path):
    global spike_list
    global amp
    global rec_dur
    global SaRa
    try:
        mat_data = scipy.io.loadmat(path)

    except:
        mat_data = h5py.File(path, 'r')
    try:
        spike_list = np.transpose(mat_data["SPIKEZ"]["TS"][0, 0])
        spike_list = np.where(np.invert(np.isnan(spike_list)), spike_list, 0)
        amp = np.zeros([spike_list.shape[1], spike_list.shape[0]])
        rec_dur = float(mat_data["SPIKEZ"][0, 0]["PREF"][0, 0]["rec_dur"][0])
        SaRa = mat_data["SPIKEZ"][0, 0]["PREF"][0, 0]["SaRa"][0][0]
        flag_mat_v1 = True
        print("Mat Datei erfolgreich importiert")
    except:
        try:
            spike_list = np.transpose(mat_data["temp"]["SPIKEZ"][0, 0]["TS"][0, 0])
            amp = mat_data["temp"]["SPIKEZ"][0, 0]["AMP"][0, 0]
            rec_dur = mat_data["temp"]["SPIKEZ"][0, 0]["PREF"][0,0]["rec_dur"][0, 0][0][0]
            SaRa = mat_data["temp"]["SPIKEZ"][0, 0]["PREF"][0, 0]["SaRa"][0, 0][0][0]
            flat_mat_v2 = True
            print("Mat Datei erfolgreich importiert")
        except:
            print("Mat Datei konnte nicht importiert werden")
    # return spike_list, amp, rec_dur, SaRa


def get_spiketimes(file, typeTs='s'):
    spkTime = np.array(file.data['3BResults/3BChEvents/SpikeTimes'])
    spkCh = np.array(file.data['3BResults/3BChEvents/SpikeChIDs'])

    if typeTs == 's':
        return spkCh, spkTime / file.sf
    elif typeTs == 'frame':
        return spkCh, spkTime


def convert_to_elephant_array_bxr(file):
    ch, ts = get_spiketimes(file)
    # array = np.zeros([ch.max() + 1, round(ts.shape[0]/ch.max())])  # 14036
    array = np.zeros([ch.max() + 1, ch.shape[0]])
    y = 0
    k = 0
    for i in ch:
        while float(array[i, k]) != 0:
            k += 1
        array[i, k] = ts[y]
        y = y + 1
        k = 0
    return array

class ReadBxr:
    def __init__(self, path):
        self.data = h5py.File(path, 'r')

        self.sf = self.data['3BRecInfo/3BRecVars/SamplingRate'][()][0]
        self.recLenght = self.data['3BRecInfo/3BRecVars/NRecFrames'][0] / self.sf

def write_bxr(data):
    pass



path_bxr = "/mnt/HDD/Data/Bxr/1min 9000Hz.bxr"
path_mat = "/mnt/HDD/Data/TS.dat/1Gy MEA101 Messung02.11.2020_07-51-55_TS.mat"
data = import_mat(path_mat)
bxr_data = ReadBxr(path_bxr)
print(bxr_data.data['3BRecInfo/3BRecVars/SamplingRate'][0])

spikes1 = []
spikes2 = []
n_spikes_einer_elektrode = np.zeros(shape=(1, spike_list.shape[0]))
for a in range(0, spike_list.shape[0]):
    # @TODO: check for zeros, so they dont have to be trimmed, takes to much time
    train = np.trim_zeros(spike_list[a]).tolist()
    n_spikes_einer_elektrode[0, a] = len(train)
    if train != []:
        for i in train:
            spikes1.append(i)
            spikes2.append(a) # Wenn Elektroden wie in Matlab bei 1 Anfangen soll muss hier a+1 -> dann müssen aber noch mehr Änderung am Code durchführt werden
vec1 =  np.array(spikes1, dtype=object)
vec2 = np.array(spikes2, dtype=object)

vec1 = vec1.astype(str).astype(float)
vec2 = vec2.astype(str).astype(int)

mea_layout = np.ones(shape=(8, 8))
mea_type = 1024
mea_ncols = 8
mea_nrows = 8
sys_chs = np.ones(shape=(1, 1))
ch2ind = np.array(bxr_data.data["3BRecInfo/3BMeaStreams/Raw/Chs"][0:456])

with h5py.File("TS2bxr.bxr", "w") as f:
    rec_info_grp = f.create_group("3BRecInfo")
    results_grp = f.create_group("3BResults")
    user_info_grp = f.create_group("3BUserInfo")

    #########################################################
    ####################3BRecInfo############################
    #########################################################

    rec_info_mea_chip = rec_info_grp.create_group("3BMeaChip")
    rec_info_mea_streams = rec_info_grp.create_group("3BMeaStreams")
    rec_info_mea_systems = rec_info_grp.create_group("3BMeaSystem")
    rec_info_rec_vars = rec_info_grp.create_group("3BRecVars")
    rec_info_source_info = rec_info_grp.create_group("3BSourceInfo")

    rec_info_mea_info_layout = rec_info_mea_chip.create_dataset("Layout", data=mea_layout, dtype='|u1')
    rec_info_mea_info_meatype = rec_info_mea_chip.create_dataset("MeaType", data=mea_type, dtype='i4')
    rec_info_mea_info_ncols = rec_info_mea_chip.create_dataset("NCols", data=mea_ncols, dtype='u4')
    rec_info_mea_info_nrows = rec_info_mea_chip.create_dataset("NRows", data=mea_nrows, dtype='u4')
    rec_info_mea_info_rois = rec_info_mea_chip.create_dataset("ROIs", (0,), dtype='|V52')
    rec_info_mea_info_syschs = rec_info_mea_chip.create_dataset("SysChs", data=sys_chs)

    rec_info_mea_streams_raw = rec_info_mea_streams.create_group("Raw")
    rec_info_mea_streams_raw_chs = rec_info_mea_streams_raw.create_dataset("Chs", data=ch2ind)

    rec_info_mea_systems_fwversion = rec_info_mea_systems.create_dataset("FwVersion", (1,), dtype='|V16')
    rec_info_mea_systems_hwversion = rec_info_mea_systems.create_dataset("HwVersion", (1,), dtype='|V16')
    rec_info_mea_systems_system = rec_info_mea_systems.create_dataset("System", (1,), dtype='i4')

    rec_info_mea_vars_bitdepth = rec_info_rec_vars.create_dataset("BitDepth", (1,), dtype='|u1')
    rec_info_mea_vars_experimenttype = rec_info_rec_vars.create_dataset("ExperimentType", (1,), dtype='i4')
    rec_info_mea_vars_maxvolt = rec_info_rec_vars.create_dataset("MaxVolt", (1,), dtype='f8')
    rec_info_mea_vars_mivolt = rec_info_rec_vars.create_dataset("MinVolt", (1,), dtype='f8')
    rec_info_mea_vars_nrecframes = rec_info_rec_vars.create_dataset("NRecFrames", (1,), dtype='i8')
    rec_info_mea_vars_samplingrate = rec_info_rec_vars.create_dataset("SamplingRate", (1,), dtype='f8')
    rec_info_mea_vars_signalinversion = rec_info_rec_vars.create_dataset("SignalInversion", (1,), dtype='f8')

    rec_info_mea_source_info_format = rec_info_source_info.create_dataset("Format", (), dtype='i4')
    rec_info_mea_source_info_guid = rec_info_source_info.create_dataset("GUID", (), dtype='|S36')
    rec_info_mea_source_info_path = rec_info_source_info.create_dataset("Path", (), dtype='|S18')

    #########################################################
    ####################3BResults############################
    #########################################################

    results_ch_events = results_grp.create_group("3BChEvents")
    results_info = results_grp.create_group("3BInfo")
    results_ch_events_lfp_ch_ids = results_ch_events.create_dataset("LfpChIDs", data=vec2)
    results_ch_events_lfp_forms = results_ch_events.create_dataset("LfpForms", (47339721,), dtype='i2')
    results_ch_events_lfp_LfpTimes = results_ch_events.create_dataset("LfpTimes", data=vec1)

    results_info_lfps = results_info.create_group("3BLFPs")

    results_info_lfps_ch_ids_2_nl_fps = results_info_lfps.create_dataset("ChIDs2NLfps", (4096,), dtype='i4')
    results_info_lfps_params = results_info_lfps.create_dataset("Params", (4096,), dtype='|V28')

    results_info_noise = results_info.create_group("3BNoise")

    results_info_noise_std_mean = results_info_noise.create_dataset("StdMean", (4090,), dtype='|V12')
    results_info_noise_valid_chs = results_info_noise.create_dataset("ValidChs", (4090,), dtype='|V4')

    results_info_cd_ids_2_labels = results_info.create_dataset("ChIDs2Labels", (4096,), dtype='|V12') # change so it runs normally: |O

    results_info_mea_chs_2_cd_ids_matrix = results_info.create_dataset("MeaChs2ChIDsMatrix", (64, 64), dtype='i4')

    results_info_mea_chs_2_cd_ids_matrix = results_info.create_dataset("MeaChs2ChIDsVector", (4090,), dtype='|V8')

    #########################################################
    ####################3BUserInfo###########################
    #########################################################

    user_info_ch_groups = user_info_grp.create_dataset("ChsGroups", (1,), dtype='|V52')
    user_info_exp_markers = user_info_grp.create_dataset("ExpMarkers", (0,), dtype='|V36')
    user_info_exp_notes = user_info_grp.create_dataset("ExpNotes", (1,), dtype='|V16')
    user_info_mea_layers_info = user_info_grp.create_dataset("MeaLayersInfo", (0,), dtype='|V92')
    user_info_time_intervals = user_info_grp.create_dataset("TimeIntervals", (1,), dtype='|V36')

    #########################################################
    #######################Writing###########################
    #########################################################






print(f.name)
test = h5py.File("mytestfile.bxr", 'r')
print(f.name)
# spike_list = convert_to_elephant_array_bxr(bxr_data)
# bxr_data.data['3BRecInfo'].keys()
# ch, ts = get_spiketimes(bxr_data)
# amp = np.zeros([spike_list.shape[1], spike_list.shape[0]])
# SaRa = bxr_data.sf
# rec_dur = bxr_data.recLenght
# bxr_data.data['3BRecInfo/3BRecVars/SamplingRate']=9000
# print(bxr_data.data['3BRecInfo/3BRecVars/SamplingRate'])
