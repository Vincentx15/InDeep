#Data processing pipeline

The processing happens in two main steps : 
 - The construction of the hdf5 database from a set of pdb files
 - The loading of this database by a Pytorch Dataloader
 
 
## The construction of the hdf5 database from a set of pdb files
There are a few steps necessary to deal with our data.
The main steps involve :
- Building a text input for the further processing
This is done in the build_database step
- Chunking the data using the cluster_peptide script, to avoid having different system sizes
- Splitting the obtained pdb into channels using the Density.Coords_channels
- Saving it into an hdf5

## The loading of the hdf5 file and use in learning
This step is less complicated and only involves two steps.
- The matrix format (n,4) is put into the grid format at the top of the Complex.py 
- The class Complex encapsulates the loading and the former transformations
into an interface that uses only the hdf5, a protein and a ligand
- Then the data is put into a Database object that is used
by a Pytorch dataloader object
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 