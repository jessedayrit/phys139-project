import sqlite3
import numpy as np


class DatabaseIO:
  # # # initialization functions # # #

  def __init__(self, dim_syndr, dim_fsyndr, n_steps_net1=20, n_steps_net2=3):
    self.dim_syndr = dim_syndr
    self.dim_fsyndr = dim_fsyndr
    self.n_steps_net1 = n_steps_net1
    self.n_steps_net2 = n_steps_net2
  # # # functions that load the databases and generate batches # # #

  def load_data(self, training_fname, validation_fname, test_fname):
    """ this function loads the training, validation, and test databases

    Input
    -----

    training_fname -- path + filename of the training database
    valiation_fname -- path + filename of the validation database
    test_fname -- path + filename of the test database
    store_in_RAM -- if True the databases will be stored in the RAM
    """

    self.training_conn = sqlite3.connect(training_fname, check_same_thread=False)
    self.validation_conn = sqlite3.connect(validation_fname, check_same_thread=False)
    self.test_conn = sqlite3.connect(test_fname, check_same_thread=False)

    training_c = self.training_conn.cursor()
    validation_c = self.validation_conn.cursor()
    test_c = self.test_conn.cursor()

    # get all the seeds
    training_c.execute('SELECT seed FROM data')
    validation_c.execute('SELECT seed FROM data')
    test_c.execute('SELECT seed FROM data')
    self.training_keys = list(
        sorted([s[0] for s in training_c.fetchall()]))
    self.validation_keys = list(
        sorted([s[0] for s in validation_c.fetchall()]))
    self.test_keys = list(sorted([s[0] for s in test_c.fetchall()]))

    # checks that there is no overlapp in the seeds of the data sets
    N_training = len(self.training_keys)
    N_validation = len(self.validation_keys)
    N_test = len(self.test_keys)
    all_keys = set(self.training_keys +
                   self.validation_keys + self.test_keys)
    if len(all_keys) < N_training + N_validation + N_test:
      raise ValueError("There is overlapp between the seeds of the"
                       " training,  validation, and test sets. This"
                       "is bad practice")
    print("loaded databases and checked exclusiveness training, "
          "validation, and test keys")

    print("N_training=" + str(N_training) + ", N_validaiton=" +
          str(N_validation) + ", N_test=" + str(N_test) + ".")

  def close_databases(self):
    """ This function closes all databases """
    self.training_conn.close()
    self.validation_conn.close()
    self.test_conn.close()

  def gen_batch(self, sample):
    """ formats a single batch of data

    Input
    -----

    sample - raw data from the database
    """

    syndr, fsyndr, parity, length = sample
    n_steps = int(len(syndr) / self.dim_syndr)

    # format into shape [steps, syndromes]
    syndr1 = np.fromstring(syndr, dtype=bool).reshape([n_steps, -1])

    # get and set length information
    len1 = np.frombuffer(length, dtype="<i4")[0]

    # the second length is set by n_steps_net2, except if len1 is shorter
    len2 = min(len1, self.n_steps_net2)

    syndr2 = syndr1[len1 - len2:len1 - len2 + self.n_steps_net2]
    fsyndr = np.fromstring(fsyndr, dtype=bool)
    parity = np.frombuffer(parity, dtype=bool)

    return syndr1, syndr2, fsyndr, len1, len2, parity

  def gen_batch_oversample(self, sample, max_steps=None):
    """ formats a single batch of data with final syndrome increments
        at each time steps into multiple batches with a signle final
        syndrome increment

    Input
    -----

    sample - raw data from the database
    max_steps -- maximum number of steps for oversampling
    """

    syndrs, fsyndrs, parities = sample
    max_steps = min([len(parities), max_steps])

    syndrs = np.fromstring(syndrs, dtype=bool).reshape([len(parities), -1])
    fsyndrs = np.fromstring(
        fsyndrs, dtype=bool).reshape([len(parities), -1])
    parities = np.frombuffer(parities, dtype=bool)

    syndr1_l, syndr2_l, fsyndr_l, len1_l, len2_l, parity_l \
        = [], [], [], [], [], []

    step_list = []
    stepsize, step = 1, 1
    for n in range(self.n_steps_net2, max_steps + 1):
      if np.mod(step, stepsize) == stepsize - 1:
        step_list.append(n)
        stepsize += 1
        step = 0
      else:
        step += 1

    for n in step_list:
      if n <= max_steps:

        # format into shape [steps, syndromes]
        syndr1 = np.concatenate(
            (syndrs[:n], np.zeros((max_steps - n, self.dim_syndr),
                                  dtype=bool)), axis=0)
        syndr1_l.append(syndr1)

        # get and set length information
        len1 = n
        len1_l.append(len1)

        # the second length is set by n_steps_net2, except if len1 is
        # shorter
        len2 = min(len1, self.n_steps_net2)
        len2_l.append(len2)

        syndr2_l.append(
            syndr1[len1 - len2:len1 - len2 + self.n_steps_net2])
        fsyndr_l.append(fsyndrs[n - 1])
        parity_l.append(parities[n - 1])

    return syndr1_l, syndr2_l, fsyndr_l, len1_l, len2_l, parity_l

  def gen_batches(self, batch_size, n_batches, data_type, oversample=False,
                  max_steps=500):
    """ a genererator to generate the training, validation, and test
        batches

    Input
    -----

    batch_size -- number of samples per batch
    n_batches -- number of batches
    data_type -- 'training', 'validation', or 'test'  data

    Output
    ------

    a generator containing formatted batches of:
    syndrome increments for the first network, for the second network,
    the final syndrome increments, length information for the first
    network, for the second network, and the final parities
    """
  
    # select data from the corresponding database
    if data_type == "training":
      c = self.training_conn.cursor()
    elif data_type == "validation":
      c = self.validation_conn.cursor()
    elif data_type == "test":
      c = self.test_conn.cursor()
    else:
      raise ValueError("The only allowed data_types are: 'training', "
                      "'validation' and 'test'.")
    
    if (data_type == "test"):
      oversample = True

    if oversample:
      c.execute("SELECT events, err_signal, parities " +
                "FROM data ORDER BY RANDOM() LIMIT ?",
                (n_batches * batch_size, ))
    else:
      c.execute("SELECT events, err_signal, parity, length " +
                "FROM data ORDER BY RANDOM() LIMIT ?",
                (n_batches * batch_size, ))

    for n in range(n_batches):

      arrX1, arrX2, arrFX, arrL1, arrL2, arrY = [], [], [], [], [], []
      samples = c.fetchmany(batch_size)

      for sample in samples:
        if oversample:
          syndr1_l, syndr2_l, fsyndr_l, len1_l, len2_l, parity_l \
              = self.gen_batch_oversample(sample, max_steps)

          for syndr1, syndr2, fsyndr, len1, len2, parity in \
                  zip(syndr1_l, syndr2_l, fsyndr_l, len1_l, len2_l, parity_l):
            arrX1.append(syndr1)
            arrX2.append(syndr2)
            arrFX.append(fsyndr)
            arrL1.append(len1)
            arrL2.append(len2)
            arrY.append(parity)
        else:
          syndr1, syndr2, fsyndr, len1, len2, parity \
              = self.gen_batch(sample)

          arrX1.append(syndr1)
          arrX2.append(syndr2)
          arrFX.append(fsyndr)
          arrL1.append(len1)
          arrL2.append(len2)
          arrY.append(parity)

      arrX1 = np.array(arrX1)
      arrX2 = np.array(arrX2)
      arrFX = np.array(arrFX)
      arrL1 = np.array(arrL1)
      arrL2 = np.array(arrL2)
      arrY = np.array(arrY)

      yield arrX1, arrX2, arrFX, arrL1, arrL2, arrY


