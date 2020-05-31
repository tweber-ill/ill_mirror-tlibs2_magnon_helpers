/**
 * tlibs2 -- helper functions in preparation for the main tlibs2 repo: https://code.ill.fr/scientific-software/takin/tlibs2
 * @author Tobias Weber <tweber@ill.fr>
 * @date 30-may-2020
 * @license GPLv3, see 'LICENSE' file
 * @desc tlibs forked on 7-Nov-2018 from the privately developed "tlibs" project (https://github.com/t-weber/tlibs).
 */

#ifndef __TLIBS2_PHYS_MAG__
#define __TLIBS2_PHYS_MAG__

#include "tlibs2/mat.h"


namespace tl2 {

/**
 * Calculates energies and dynamical structure factors from Landau-Lifshitz (M x) and fluctuation matrices.
 * Uses the mathematical formalism by M. Garst et al., references:
 *	- https://doi.org/10.1088/1361-6463/aa7573
 *  - https://doi.org/10.1038/nmat4223 (supplement)
 *  - https://kups.ub.uni-koeln.de/7937/
 */
template<class t_mat_cplx, class t_vec_cplx, class t_cplx, class t_real>
std::tuple<std::vector<t_cplx>, std::vector<t_vec_cplx>, std::vector<t_mat_cplx>>
calc_dynstrucfact_landau(const t_mat_cplx& Mx, const t_mat_cplx& Fluc,
	t_real normfac=1, const t_real* mineval = nullptr, const t_real *maxeval = nullptr,
	std::size_t MxsubMatSize=3, std::size_t MxsubMatRowBegin=0, t_real eps=1e-6)
{
	constexpr t_cplx imag = t_cplx(0,1);

	// calculate Mx eigenvalues
	std::vector<t_vec_cplx> Mxevecs;
	std::vector<t_cplx> Mxevals;
	{
		std::vector<t_real> _Mxevals;
		//if(!eigenvec_cplx<t_real>(Mx, Mxevecs, Mxevals, true))
		if(!eigenvecsel_herm<t_real>(-imag*Mx, Mxevecs, _Mxevals, true, -1., -2., eps))
			throw std::runtime_error("Mx eigenvector determination failed!");
		for(t_real d : _Mxevals) Mxevals.emplace_back(t_cplx(0., d));


		// filter eigenvalues
		auto maxelem = std::max_element(_Mxevals.begin(), _Mxevals.end(),
			[](t_real x, t_real y) -> bool { return std::abs(x) < std::abs(y); });

		std::vector<t_vec_cplx> Mxevecs_new;
		std::vector<t_cplx> Mxevals_new;

		for(std::size_t elem=0; elem<_Mxevals.size(); ++elem)
		{
			// upper eigenvalue limit
			if(maxeval && std::abs(_Mxevals[elem]) < std::abs(*maxelem)**maxeval)
				continue;
			// lower eigenvalue limit
			if(mineval && std::abs(_Mxevals[elem]) < *mineval)
				continue;
			Mxevecs_new.push_back(Mxevecs[elem]);
			Mxevals_new.push_back(Mxevals[elem]);
		}

		Mxevecs = std::move(Mxevecs_new);
		Mxevals = std::move(Mxevals_new);
	}


	// convert to eigenvector matrix
	t_mat_cplx MxEvecs(Mx.size1(), Mxevecs.size());
	for(std::size_t idx1=0; idx1<MxEvecs.size1(); ++idx1)
		for(std::size_t idx2=0; idx2<MxEvecs.size2(); ++idx2)
			MxEvecs(idx1, idx2) = Mxevecs[idx2][idx1];

	t_mat_cplx MxEvecsH = hermitian(MxEvecs);
	t_mat_cplx MxEvecs3 = submatrix_wnd<t_mat_cplx>(MxEvecs, MxsubMatSize, Mxevecs.size(), MxsubMatRowBegin, 0);
	t_mat_cplx MxEvecsH3 = hermitian(MxEvecs3);

	// transform fluctuation matrix into Mx eigenvector system
	t_mat_cplx invsuscept = prod_mm(Fluc, MxEvecs);
	invsuscept = prod_mm(MxEvecsH, invsuscept);

	// transform Mx into Mx eigenvector system
	// Mxx is diagonal with this construction => directly use Mxevals
	//t_mat_cplx Mxx = prod_mm(Mx, MxEvecs);
	//Mxx = prod_mm(MxEvecsH, Mxx);
	t_mat_cplx Mxx = diag_matrix<t_mat_cplx>(Mxevals);
	Mxx *= imag / normfac;

	// Landau-Lifshitz: d/dt dM = -Mx B_mean, B_mean = -chi^(-1) * dM
	// E = EVals{ i Mx chi^(-1) }
	// chi_dyn^(-1) = i*E*Mx^(-1) + chi^(-1)
	// Mx*chi_dyn^(-1) = i*E + Mx*chi^(-1)
	t_mat_cplx Interactmat = prod_mm(Mxx, invsuscept);
	std::vector<t_vec_cplx> Interactevecs;
	std::vector<t_cplx> Interactevals;
	if(!eigenvec_cplx<t_real>(Interactmat, Interactevecs, Interactevals, true))
		throw std::runtime_error("Mxx eigenvector determination failed!");

	std::vector<t_mat_cplx> Interactemats;
	Interactemats.reserve(Interactevals.size());

	for(std::size_t iInteract=0; iInteract<Interactevals.size(); ++iInteract)
	{
		const t_cplx& eval = Interactevals[iInteract];
		const t_vec_cplx& evec = Interactevecs[iInteract];

		auto evec_scale = prod_mv(Mxx, evec);

		auto matOuter = outer_cplx<t_vec_cplx, t_mat_cplx>(evec, evec);
		matOuter /= inner_cplx<t_vec_cplx>(evec, evec_scale);

		t_mat_cplx emat = prod_mm(matOuter, MxEvecsH3);
		emat = prod_mm(MxEvecs3, emat);
		Interactemats.emplace_back(std::move(emat));
		//eigs.emplace_back(Eig{.eval=eval, .evec=evec, .emat=emat});
	}

	return std::make_tuple(Interactevals, Interactevecs, Interactemats);
}



/**
 * Gets the dynamical structure factors from the eigenvectors calculated using calc_dynstrucfact_landau.
 */
template<class t_mat_cplx, class t_vec_cplx, class t_cplx, class t_real>
std::tuple<t_real, std::vector<t_real>>
get_dynstrucfact_neutron(
	const t_cplx& eval, const t_vec_cplx& evec, const t_mat_cplx& _emat,
	const t_mat_cplx* projNeutron=nullptr, const std::vector<t_mat_cplx>* pol = nullptr)
{
	t_real E = eval.real();
	t_mat_cplx emat = _emat;

	// magnetic neutron scattering orthogonal projector: projNeutron = 1 - |G><G|
	if(projNeutron)
	{
		emat = prod_mm(emat, *projNeutron);
		emat = prod_mm(*projNeutron, emat);
	}

	std::vector<t_real> sfacts;

	// unpolarised structure factor
	sfacts.push_back(std::abs(trace(emat).real()));

	// polarised structure factors
	if(pol)
	{
		for(const t_mat_cplx& polmat : *pol)
		{
			t_mat_cplx matSF = prod_mm(polmat, emat);
			sfacts.push_back(std::abs(trace(matSF).real()));
		}
	}

	return std::make_tuple(E, sfacts);
}

}
#endif
