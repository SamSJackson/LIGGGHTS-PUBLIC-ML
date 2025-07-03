/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    This file is from LAMMPS, but has been modified. Copyright for
    modification:

    Copyright 2012-     DCS Computing GmbH, Linz
    Copyright 2009-2012 JKU Linz

    Copyright of original file:
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- */

#include <cmath>
#include <string.h>
#include <stdlib.h>
#include "compute_contact_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

// sam:
#ifdef SUPERQUADRIC_ACTIVE_FLAG
  #include "math_extra_liggghts_superquadric.h" 
  #include <iostream>
#endif 

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeContactAtom::ComputeContactAtom(LAMMPS *lmp, int &iarg, int narg, char **arg) :
  Compute(lmp, iarg, narg, arg)
{
  
  if (narg < iarg) error->all(FLERR,"Illegal compute contact/atom command");

  skin = 0.;

  if(narg > iarg)
  {
      if (narg < iarg+2)
          error->all(FLERR,"Illegal compute contact/atom command");
      if(strcmp("skin",arg[iarg++]))
          error->all(FLERR,"Illegal compute contact/atom command, expecting keyword 'skin'");
      skin = atof(arg[iarg++]);
  }
  
  peratom_flag = 1;
  size_peratom_cols = 0;
  comm_reverse = 1;

  nmax = 0;
  contact = NULL;

  // error checks

  if (!atom->sphere_flag && !atom->superquadric_flag)
    error->all(FLERR,"Compute contact/atom requires atom style sphere or atom style superquadric!");
}

/* ---------------------------------------------------------------------- */

ComputeContactAtom::~ComputeContactAtom()
{
  memory->destroy(contact);
}

/* ---------------------------------------------------------------------- */

void ComputeContactAtom::init()
{
  if (force->pair == NULL)
    error->all(FLERR,"Compute contact/atom requires a pair style be defined");

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"contact/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute contact/atom");

  // need an occasional neighbor list

  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->gran = 1;
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeContactAtom::init_list(int id, NeighList *ptr)
{
  list = ptr;
  
}

/* ---------------------------------------------------------------------- */
// sam: update this function

#ifdef SUPERQUADRIC_ACTIVE_FLAG
void ComputeContactAtom::compute_peratom()
{
	int *ilist, *jlist, *numneigh, **firstneigh;
	int inum, jnum;
	double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
	double radi, radsum, radsumsq;

	// narrow-phase variables
	double shapeA[3], shapeB[3], blockA[2], blockB[2], quatA[4], quatB[4];
	double xi[3], xj[3], cpoint[3];
	double fi, fj; 
	bool fail;
	
	invoked_peratom = update->ntimestep;

	if (atom->nmax > nmax) {
		memory->destroy(contact);
		nmax = atom->nmax;
		memory->create(contact,nmax,"contact/atom:contact");
		vector_atom = contact;
	}

	neighbor->build_one(list->index);

	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	// sam-TODO: Do I need to include nghost?
	// Need to determine if this is narrow/broad detection

	// sam-NOTE: atom is the class object, not necessarily one atom 
	double **x = atom->x;
	double *radius = atom->radius;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;

	// narrow-phase variables
	double **shape = atom->shape;
	double **blockiness = atom->blockiness;
	double **quaternion = atom->quaternion;

	Superquadric particleA;
	Superquadric particleB;

	for (int i=0; i < nall; i++) contact[i] = 0.0;

	// investigate interactions of every pair
	for (int ii=0; ii < inum; ii++) {
		fail = false;
		int i = ilist[ii];
		if (mask[i] & groupbit) {
			jlist = firstneigh[i];
			jnum = numneigh[i];

			xtmp = x[i][0];	
			xtmp = x[i][1];	
			xtmp = x[i][2];	
			radi = radius[i];

			for (int jj=0; jj < jnum; jj++) {
				int j = jlist[jj];
				j &= NEIGHMASK;
	
				// broad-phase
				delx = xtmp - x[j][0];
				dely = ytmp - x[j][1];
				delz = ztmp - x[j][2];
				rsq = delx*delx + dely*dely + delz*delz;
				radsum = radi + radius[j] + skin;
				radsumsq = radsum * radsum;
				// if min bounding spheres don't overlap, skip
				if (rsq > radsumsq) continue;
				// narrow-phase

				// TODO: move this into superquadric.cpp as initialiser 
				shapeA[0] = shape[i][0];
				shapeA[1] = shape[i][1];
				shapeA[2] = shape[i][2];

				shapeB[0] = shape[j][0];
				shapeB[1] = shape[j][1];
				shapeB[2] = shape[j][2];

				blockA[0] = blockiness[i][0];
				blockA[1] = blockiness[i][1];

				blockB[0] = blockiness[j][0];
				blockB[1] = blockiness[j][1];

				quatA[0] = quaternion[i][0];
				quatA[1] = quaternion[i][1];
				quatA[2] = quaternion[i][2];
				quatA[3] = quaternion[i][3];

				quatB[0] = quaternion[j][0];
				quatB[1] = quaternion[j][1];
				quatB[2] = quaternion[j][2];
				quatB[3] = quaternion[j][3];

				xi[0] = x[i][0];
				xi[1] = x[i][1];
				xi[2] = x[i][2];

				xj[0] = x[j][0];
				xj[1] = x[j][1];
				xj[2] = x[j][2];
				// move above into superquadric.cpp 

				particleA = Superquadric(xi, quatA, shapeA, blockA);
				particleB = Superquadric(xj, quatB, shapeB, blockB);

				fail = MathExtraLiggghtsNonspherical::calc_contact_point_if_no_previous_point_available(
						&particleA,
						&particleB,
						cpoint,
						fi,
						fj,
						error  
				);

				fprintf(screen, "Completed this: %d\n", fail);

				if (fail) {
					contact[i] += 1.0;
					contact[j] += 1.0;
				}
			}
		}
	}
}

#else
void ComputeContactAtom::compute_peratom()
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double radi,radsum,radsumsq;
  int *ilist,*jlist,*numneigh,**firstneigh;

  invoked_peratom = update->ntimestep;

  // grow contact array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(contact);
    nmax = atom->nmax;
    memory->create(contact,nmax,"contact/atom:contact");
    vector_atom = contact;
  }

  // invoke neighbor list (will copy or build if necessary)

  neighbor->build_one(list->index);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute number of contacts for each atom in group
  // contact if distance <= sum of radii
  // tally for both I and J

  double **x = atom->x;
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  for (i = 0; i < nall; i++) contact[i] = 0.0;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      radi = radius[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        radsum = radi + radius[j] + skin; 
        radsumsq = radsum*radsum;
        if (rsq <= radsumsq) {
          contact[i] += 1.0;
          contact[j] += 1.0;
        }
      }
    }
  }

  // communicate ghost atom counts between neighbor procs if necessary

  if (force->newton_pair) comm->reverse_comm_compute(this);
}
#endif

/* ---------------------------------------------------------------------- */

int ComputeContactAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
    buf[m++] = contact[i];
  return 1;
}

/* ---------------------------------------------------------------------- */

void ComputeContactAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    contact[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeContactAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
