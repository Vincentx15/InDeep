from __future__ import absolute_import
from __future__ import print_function
import sys
import os
from time import perf_counter

# entry point to PyMOL's API
from pymol import cmd

# pymol.Qt provides the PyQt5 interface, but may support PyQt4
# and/or PySide as well
from pymol.Qt import QtWidgets

from pymol.Qt.utils import loadUi

import numpy as np
import shutil
import hashlib

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../..'))

from learning import utils as lutils, predict, model_factory
from data_processing import utils
from post_processing import blobber, Highlighter


def __init_plugin__(app=None):
    """
    Add an entry to the PyMOL "Plugin" menu
    """
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('deepPPI Plugin', run_plugin_gui)


# global reference to avoid garbage collection of our dialog
dialog_object = None


def run_plugin_gui():
    '''
    Open our custom dialog
    '''
    global dialog_object

    if dialog_object is None:
        dialog_object = Dialog()

    dialog_object.dialog.show()


class Dialog:
    def __init__(self):
        # Set static values
        # cmd.fetch('1ycr')
        self.user_npz = False
        cmd.set('normalize_ccp4_maps', 0)
        self.spacing = 1
        self.padding = 8
        self.range_value = 1000
        self.experiments_file_path = os.path.join(script_dir,
                                                  '../../results/experiments')
        self.atomtypes = utils.read_atomtypes()
        # create a new Window
        self.dialog = QtWidgets.QDialog()
        # self.dialog = QFileDialog()

        # populate the Window from our *.ui file which was created with
        # the Qt Designer using os.listdir
        uifile = os.path.join(os.path.dirname(__file__), 'gui_highlight.ui')
        self.form = loadUi(uifile, self.dialog)
        explist = os.listdir(self.experiments_file_path)
        self.form.model_choice.insertItems(0, explist)

        # Create the bindings with all buttons
        self.bindings()

        # placeholders
        self.save_path = None

        # try creating a results file and do a results/
        self.saving_path = os.path.join(os.path.dirname(__file__),
                                        '../../results/pymol_preds')
        try:
            os.mkdir(self.saving_path)
        except FileExistsError:
            pass

        self.temp_path = os.path.join(os.path.dirname(__file__), 'temp_dir')
        try:
            os.mkdir(self.temp_path)
        except FileExistsError:
            pass

        self.highlight_recent = 0
        self.highlight_hd = True

    def __del__(self):
        print('deleted', self.temp_path)
        os.removedirs(self.temp_path)

    def clean(self):
        print('deleted', self.temp_path)
        shutil.rmtree(self.temp_path)

    @property
    def experiment(self):
        return self.form.model_choice.currentText()

    @property
    def experiments_path(self):
        return os.path.join(self.experiments_file_path, self.experiment)

    @property
    def selection(self):
        return self.form.selection_input.text()

    @property
    def npz_path(self):
        if not self.user_npz:
            return os.path.join(self.saving_path, self.pdb_hash + '_'
                                + self.experiments_name + '.npz')
        else:
            return self.form.lineEdit_npz.text()

    @property
    def experiments_name(self):
        # Remove the .exp from the name to only get the name
        return self.experiment[:-4]

    @property
    def pdb_hash(self):
        # Remove the .exp from the name to only get the name
        string_name = open(os.path.join(self.temp_path, 'current_prot.pdb')).read()
        string_name = string_name.encode('utf-8')
        return hashlib.md5(string_name).hexdigest()

    def get_volume_hd(self, isovalue):
        return (self.pocket_hd > isovalue).sum()

    def get_volume_pl(self, isovalue):
        return (self.pocket_pl > isovalue).sum()

    def load_model(self):
        print('Loading model: %s' % self.experiment)
        model = model_factory.model_from_exp(expfilename=self.experiments_path, load_weights=True)
        return model

    def save_selection(self):
        user_selection = self.selection
        cmd.select('input_sel', selection=user_selection)
        cmd.color('red', 'input_sel')
        cmd.color('cyan', '(byobject input_sel) and not input_sel')
        cmd.disable('input_sel')
        cmd.save(os.path.join(self.temp_path, 'current_prot.pdb'),
                 selection=user_selection)

    def load_values(self, hd=True):
        with np.load(self.npz_path, allow_pickle=True) as npzfile:
            origin = npzfile['origin']

            if hd:
                grid = npzfile['hd']
                all_coords = npzfile['hd_coords']
                all_distribs = npzfile['hd_distribs']
                all_ids = npzfile['hd_ids']
            else:
                grid = npzfile['pl']
                all_coords = npzfile['pl_coords']
                all_distribs = npzfile['pl_distribs']
                all_ids = npzfile['pl_ids']
        return grid, all_coords, all_distribs, all_ids, origin

    def get_mrc(self, blob_id=1, channel_id=0, HD=True, fill_combo=False, first_predict=False):
        """
        Convert npzfile to mrc
        Create the appropriate slider and values
        """

        grid, all_coords, all_distribs, all_ids, origin = self.load_values(HD)

        try:
            iter(all_coords)
        except TypeError:
            return False

        if HD:
            grid = np.squeeze(grid)
            if first_predict:
                channel_id = grid.shape[0] - 1
                self.form.channelBox.clear()
                self.form.channelBox.addItems(list(self.atomtypes.keys()) + ['ALL'])
                self.form.channelBox.setCurrentIndex(channel_id)

            void_channel = (channel_id == grid.shape[0] - 1)
            grid = grid[channel_id]
            if void_channel:
                grid = 1. - grid

        _, id_grid = lutils.dense(grid, all_coords,
                                               all_distribs,
                                               all_ids)
        # blob_label = np.unique(id_grid)[blob_label]

        extracted = grid * (id_grid == blob_id).astype(np.int)
        if HD:
            self.pocket_hd = extracted
        else:
            self.pocket_pl = extracted
        isomin = np.min(extracted[extracted > 0.])
        isomax = np.max(extracted)
        # isomin = (extracted[extracted > 0.]).min()
        # isomax = extracted.max()

        if HD:
            # Add the prediction to the pymol session and display it at the
            # lowest isovalue
            # Construct the slider and adjust the axis TODO : fixme
            self.form.isomin_ppi.setText('%.3f' % isomin)
            self.form.isomax_ppi.setText('%.3f' % isomax)
            self.form.ppiSlider.setMinimum(int(isomin * self.range_value))
            self.form.ppiSlider.setMaximum(int(isomax * self.range_value))
            self.form.ppiSlider.setTickInterval(1)

            # Get pocket list and construct combobox accordingly
            n_pockets_ppi = np.max(all_ids)
            if fill_combo:
                self.form.ppi_comboBox.clear()
                self.form.ppi_comboBox.addItems([str(i + 1)
                                                 for i in
                                                 range(n_pockets_ppi)])
            try:
                sorted_array = np.sort(all_distribs[all_ids == blob_id])
                score = blobber.score_blob(sorted_array)
            except:
                score = 0
            self.form.score_ppi.setText(f'Pocket score : {score:.3f}')
            utils.save_density(extracted,
                               os.path.join(self.temp_path, 'grid_ppi.mrc'),
                               self.spacing, origin,
                               self.padding)
            cmd.delete('mrc_ppi')
            cmd.delete('surf_ppi')
            cmd.load(os.path.join(self.temp_path, 'grid_ppi.mrc'), 'mrc_ppi')
            self.update_all(isomin)

        else:
            # Add the prediction to the pymol session and display it at the
            # lowest isovalue
            # Construct the slider and adjust the axis TODO : fixme
            self.form.isomin_pl.setText('%.3f' % isomin)
            self.form.isomax_pl.setText('%.3f' % isomax)
            self.form.plSlider.setMinimum(int(isomin * self.range_value))
            self.form.plSlider.setMaximum(int(isomax * self.range_value))
            self.form.plSlider.setTickInterval(1)
            n_pockets_pl = np.max(all_ids)
            if fill_combo:
                self.form.pl_comboBox.clear()
                self.form.pl_comboBox.addItems([str(i + 1)
                                                for i in range(n_pockets_pl)])

            try:
                sorted_array = np.sort(all_distribs[all_ids == blob_id])
                score = blobber.score_blob(sorted_array)
            except:
                score = 0
            self.form.score_pl.setText(f'Pocket score : {score:.3f}')
            utils.save_density(extracted,
                               os.path.join(self.temp_path, 'grid_pl.mrc'),
                               self.spacing, origin,
                               self.padding)
            cmd.delete('mrc_pl')
            cmd.delete('surf_pl')
            cmd.load(os.path.join(self.temp_path, 'grid_pl.mrc'), 'mrc_pl')
            self.update_all(isomin, ppi=False)
        return True

    def predict(self):
        self.user_npz = False
        msg = 'Status: Predicting on %s with model %s. Please wait ...' % (self.selection, self.experiment)
        self.form.status.setText(msg)
        self.form.status.repaint()
        self.save_selection()
        cmd.save(os.path.join(self.temp_path, 'current_session.pse'))
        try:
            np.load(self.npz_path)
        except FileNotFoundError:
            model = self.load_model()
            pdbname = os.path.join(self.temp_path, 'current_prot.pdb')
            out_hd, out_pl, origin, _ = predict.predict_pdb(model,
                                                            pdbname,
                                                            spacing=self.spacing,
                                                            padding=self.padding)
            cmd.load(os.path.join(self.temp_path, 'current_session.pse'))
            # Watershed on HD
            grid = np.squeeze(out_hd)
            grid = grid[-1]
            grid = 1. - grid
            # utils.save_density(grid, 'raw.mrc', spacing, origin, padding)
            hd_coords, hd_distribs, hd_ids = blobber.to_blobs(grid, hetatm=False)

            name = "pymol"
            mrcfilename = f"{name}_ALL.mrc"
            utils.save_prediction(pred=out_hd,
                                  name=name,
                                  spacing=self.spacing,
                                  origin=origin,
                                  padding=self.padding)
            cmd.load(filename=mrcfilename)

            # Watershed on PL
            grid = out_pl.squeeze()
            utils.save_density(grid, 'PL_ALL.mrc',
                               self.spacing, origin, self.padding)
            cmd.load(filename='PL_ALL.mrc')

            pl_coords, pl_distribs, pl_ids = blobber.to_blobs(grid, hetatm=True)
            np.savez_compressed(self.npz_path,
                                hd=np.squeeze(out_hd),
                                pl=np.squeeze(out_pl),
                                origin=origin,
                                hd_coords=hd_coords,
                                hd_distribs=hd_distribs,
                                hd_ids=hd_ids,
                                pl_coords=pl_coords,
                                pl_distribs=pl_distribs,
                                pl_ids=pl_ids)
        finally:
            success_hd = self.get_mrc(blob_id=1, fill_combo=True, first_predict=True)
            success_pl = self.get_mrc(blob_id=1, HD=False, fill_combo=True)
            msg = 'Status: Prediction done on %s with model %s. \n' % (self.selection, self.experiment)
            if success_hd and success_pl:
                apd = "Found pockets for PL and HD"
            elif success_hd:
                apd = "Found pockets only for HD"
            elif success_pl:
                apd = "Found pockets only for PL"
            else:
                apd = "Found no pockets for PL or HD"
            msg = msg + apd
            self.form.status.setText(msg)
            self.form.update()

    def update_all(self, isovalue, ppi=True):
        # This is to take only the first range value decimals
        isovalue = float(int(isovalue * self.range_value)) / self.range_value
        if ppi:
            cmd.isosurface('surf_ppi', 'mrc_ppi', isovalue)
            cmd.color('yellow', 'surf_ppi')
            cmd.enable('surf_ppi')
            self.form.current_iso_ppi.setText(str(isovalue))
            self.form.ppiSlider.setValue(int(isovalue * self.range_value))
            volume = self.get_volume_hd(isovalue)
            self.form.ppi_volume.setText('Volume: %d Å³' % volume)
        else:
            cmd.isosurface('surf_pl', 'mrc_pl', isovalue)
            cmd.color('green', 'surf_pl')
            cmd.enable('surf_pl')
            self.form.current_iso_pl.setText(str(isovalue))
            self.form.plSlider.setValue(int(isovalue * self.range_value))
            volume = self.get_volume_pl(isovalue)
            self.form.pl_volume.setText('Volume: %d Å³' % volume)

    def set_highlight(self, hd=True):
        self.highlight_hd = hd
        self.highlight_recent = perf_counter()
        msg = 'Status: Highlighting is computationnally intensive and will take several minutes. \n' \
              'Please press "Validate" in the following 10 seconds to confirm your choice'
        self.form.status.setText(msg)
        self.form.status.repaint()

    def highlight(self):
        if perf_counter() - self.highlight_recent > 10:
            return
        try:
            int(self.form.ppi_comboBox.currentText())
        except ValueError:
            print('You need to do a prediction and select a blob to highlight it')
            return
        msg = 'Status: Highlighting '
        self.form.status.setText(msg)
        self.form.status.repaint()
        highlight = Highlighter.Highlighter(experiments_name=self.experiments_name, hd=self.highlight_hd)
        highlight.mutate_experiment(blob_id=int(self.form.ppi_comboBox.currentText()), radius=6)
        cmd.load(filename=highlight.dump_path)
        del highlight
        msg = 'Status: Prediction done '
        self.form.status.setText(msg)
        self.form.status.repaint()

    def select_npz(self):
        npzfilename = QtWidgets.QFileDialog.getOpenFileName()[0]
        print(f"npz file: {npzfilename}")
        self.form.lineEdit_npz.setText(npzfilename)
        self.user_npz = True

    def load_npz(self):
        npzfilename = self.form.lineEdit_npz.text()
        print(f"Loading: {npzfilename}")
        self.get_mrc(blob_id=1, fill_combo=True, first_predict=True)
        self.get_mrc(blob_id=1, HD=False, fill_combo=True)

    def bindings(self):
        # Validate the choice of the selection
        self.form.selectButton.clicked.connect(self.save_selection)
        # self.form.selection_buttonBox.accepted.connect(self.save_selection)

        # Launch a prediction on the selection
        self.form.predictButton.clicked.connect(self.predict)

        # Load a npz file
        self.form.select_npz.clicked.connect(self.select_npz)
        self.form.pushButton_load_npz.clicked.connect(self.load_npz)

        # PPI BINDINGS
        #################

        # Choose the mrc signal/pocket to display
        self.form.ppi_comboBox.activated.connect(lambda: self.get_mrc(blob_id=int(self.form.ppi_comboBox.currentText()),
                                                                      channel_id=int(
                                                                          self.form.channelBox.currentIndex())))

        # Choose the channel to display
        self.form.channelBox.activated.connect(lambda: self.get_mrc(blob_id=int(self.form.ppi_comboBox.currentText()),
                                                                    channel_id=int(
                                                                        self.form.channelBox.currentIndex())))

        # Update values displayed when slider is moved
        self.form.ppiSlider.sliderMoved.connect(
            lambda: self.update_all(self.form.ppiSlider.sliderPosition() / self.range_value))

        # Update values displayed when the value is changed
        self.form.current_iso_ppi.editingFinished.connect(
            lambda: self.update_all(float(self.form.current_iso_ppi.text())))

        # PL BINDINGS
        #################
        # Choose the mrc signal/pocket to display
        self.form.pl_comboBox.activated.connect(
            lambda: self.get_mrc(int(self.form.pl_comboBox.currentText()),
                                 HD=False))

        # Update values displayed when slider is moved
        self.form.plSlider.sliderMoved.connect(
            lambda: self.update_all(self.form.plSlider.sliderPosition() / self.range_value, ppi=False))

        # Update values displayed when the value is changed
        self.form.current_iso_pl.editingFinished.connect(
            lambda: self.update_all(float(self.form.current_iso_pl.text()), ppi=False))

        # self.form.highlight_ppi.clicked.connect(lambda: self.set_highlight(hd=True))
        # self.form.highlight_pl.clicked.connect(lambda: self.set_highlight(hd=False))
        # self.form.validate.clicked.connect(self.highlight)
