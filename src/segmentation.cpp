#include"flowsac.h"

#include"CVX/core.h"
#include"CVX/cc.h"
using namespace cv;

#include<list>

//create region from UCM with a given threshold, region[i]==0 represents an edge point
//return the number of regions (include the region of edges)
inline int ucm2Region(const Mat1f &ucm, float threshold, Mat1i &region)
{
	Mat1b edgeMask(ucm.size());
	for_each_2(DWHN1(ucm), DN1(edgeMask), [threshold](float v, uchar &m) {
		m = v > threshold ? 255 : 0;
	});
	region.create(edgeMask.size());
	int nregion = cc_seg_n4c1(DWHN1(edgeMask), DN(region));

	//set region index of edges to 0
	std::unique_ptr<uchar[]> _isEdge(new uchar[nregion]);
	uchar *isEdge = _isEdge.get();
	memset(isEdge, 0, nregion);

	for_each_2(DWHN1(edgeMask), DN1(region), [isEdge](uchar m, int i) {
		if (m)
			isEdge[i] = 1;
	});

	std::unique_ptr<int[]> _remap(new int[nregion]);
	int *remap = _remap.get();

	int id = 1; // 0 is edge
	for (int i = 0; i < nregion; ++i)
	{
		remap[i] = isEdge[i] ? 0 : id++;
	}
	for_each_1(DWHN1(region), [remap](int &i) {
		i = remap[i];
	});
	return id;
}

class HSeg
{
public:
	class RegionEdgeBase
	{
	public:
		int ri, rj; //ri<rj

		RegionEdgeBase(int _ri = 0, int _rj = 0)
			:ri(_ri), rj(_rj)
		{}
	};
	class RegionEdge
		:public RegionEdgeBase
	{
	public:
		float w;	//edge weight;
		int   len;  //length
	public:
		RegionEdge(int _ri = 0, int _rj = 0)
			:RegionEdgeBase(_ri, _rj), w(0), len(0)
		{}
		friend bool operator<(const RegionEdge &a, const RegionEdge &b)
		{
			return a.w < b.w;
		}
	};
	static void searchRegionEdges(const Im32SC1 &region, const Im32FC1 &edgeWeight, std::list<RegionEdge> &vedges, int maxNBR = 10)
	{
		int nregion = maxElem(region) + 1;
		struct Region
		{
			RegionEdge *nbr;
			int  nnbr;
			int  capacity;
		public:
			int find(int rj)
			{
				for (int i = nnbr - 1; i >= 0; --i) //search reverse for acceleration
				{
					if (nbr[i].rj == rj)
						return i;
				}
				return -1;
			}
			int addEdge(int ri, int rj, float e, int initCapacity)
			{
				int k = find(rj);
				if (k<0)
				{
					if (nnbr >= capacity)
					{
						int  _capacity = nnbr * 2;
						RegionEdge *_nbr = new RegionEdge[_capacity];
						memcpy(_nbr, nbr, sizeof(nbr[0])*nnbr);
						if (capacity > initCapacity)
							delete[]nbr;
						nbr = _nbr;
						capacity = _capacity;
					}
					nbr[nnbr] = RegionEdge(ri, rj);
					k = nnbr++;
				}

				nbr[k].w += e;
				nbr[k].len++;

				return k;
			}
		};
		std::unique_ptr<Region[]> _r(new Region[nregion]);
		Region *r = _r.get();

		std::unique_ptr<RegionEdge[]> _buffer(new RegionEdge[nregion*maxNBR]);
		for (int i = 0; i < nregion; ++i)
		{
			r[i].nbr = _buffer.get() + i*maxNBR;
			r[i].nnbr = 0;
			r[i].capacity = maxNBR;
		}

		auto addEdge = [r, maxNBR](int ri, int rj, float e) {
			if (ri > rj)
				std::swap(ri, rj);
			if (ri != 0)
				r[ri].addEdge(ri, rj, (float)e, maxNBR);
		};

		Rect roi(1, 1, region.cols - 2, region.rows - 2);
		const int rstride = stepC(region);

		for_each_2(DWHNCr(region, roi), DN1r(edgeWeight, roi), [addEdge, rstride](const int *p, float e) {
			if (*p == 0) //if is an edge point
			{
				int ri = p[-1], rj = p[1];
				if (ri != rj)
					addEdge(ri, rj, e);

				ri = p[-rstride], rj = p[rstride];
				if (ri != rj)
					addEdge(ri, rj, e);
			}
		});

		//vedges.reserve(maxNBR*nregion);
		for (int i = 1; i < nregion; ++i)
		{
			for (int j = 0; j < r[i].nnbr; ++j)
				vedges.push_back(r[i].nbr[j]);

			if (r[i].capacity>maxNBR) //if has reallocated
				delete[]r[i].nbr;
		}
	}
public:
	class RegionNode
	{
	public:
		std::vector<int> elem;
		int				 size;
		int				 msize;

		void merge(const RegionNode &r)
		{
			for (size_t i = 0; i < r.elem.size(); ++i)
				elem.push_back(r.elem[i]);
			size += r.size;
			msize += r.msize;
		}
		void clear()
		{
			elem.clear();
			size = 0;
			msize = 0;
		}
	};
	std::vector<RegionNode>			regions;
	std::list<RegionEdge>			edges;
	std::vector<RegionEdge>			edgesSorted;
	std::vector<RegionEdgeBase>     veq;
	Im32SC1                         initRegion;
	int								initNCC;
	int								largeRegionStart = 0;
private:
	void _initSPGraph(const Im32SC1 &region, const Im32FC1 &edgeWeight, const Im8UC1 &sizeMask)
	{
		int nr = maxElem(region) + 1;
		initNCC = nr;

		regions.clear();
		regions.resize(nr);
		for (int i = 0; i < nr; ++i)
		{
			regions[i].elem.push_back(i);
			regions[i].size = 0;
			regions[i].msize = 0;
		}
		{//get region size
			std::vector<RegionNode> &_regions(regions);
			//if (sizeMask.empty())
			for_each_1(DWHN1(region), [&_regions](int i) {
				_regions[i].size++;
			});
			//else
			for_each_2(DWHN1(region), DN1(sizeMask), [&_regions](int i, uchar m) {
				if (m) {
					_regions[i].msize++;
				}

			});
		}

		searchRegionEdges(region, edgeWeight, edges);
		for (auto &e : edges)
		{
			if (e.len > 0)
				e.w /= e.len; //compute average
		}

		edges.sort();
	}

	typedef std::list<RegionEdge>::iterator _ItrT;
	bool _removeEdge(_ItrT eitr)
	{
		auto e0(*eitr);
		veq.push_back(RegionEdgeBase(regions[e0.ri].elem[0], regions[e0.rj].elem[0]));
		edgesSorted.push_back(e0);

		regions[e0.ri].merge(regions[e0.rj]);
		regions[e0.rj].clear(); //mark as merged.

		edges.erase(eitr);

		_ItrT enull = edges.end();
		std::vector<_ItrT> vindex(regions.size(), enull);
		std::vector<bool>  vmerged(regions.size(), false);
		std::vector<_ItrT> merged;
		for (auto itr = edges.begin(); itr != edges.end();)
		{
			auto &e(*itr);
			bool bdel = false;

			if (e.ri == e0.rj)
				e.ri = e0.ri;
			if (e.rj == e0.rj)
				e.rj = e0.ri;

			if (e.ri == e0.ri || e.rj == e0.ri)
			{
				if (e.ri > e.rj)
					std::swap(e.ri, e.rj);
				int spn = e.ri == e0.ri ? e.rj : e.ri;
				if (vindex[spn] != enull)
				{
					auto mitr = vindex[spn];
					mitr->w = (mitr->w*mitr->len + e.w*e.len) / (mitr->len + e.len);
					mitr->len += e.len;
					if (!vmerged[spn])
					{
						merged.push_back(mitr);
						vmerged[spn] = true;
					}
					auto ditr = itr;
					++itr;
					edges.erase(ditr);
					bdel = true;
				}
				else
					vindex[spn] = itr;
			}
			if (!bdel)
				++itr;
		}
		//re-sort the merged edges
		//edges.sort();
		for (_ItrT itr : merged)
		{
			RegionEdge e(*itr);
			_ItrT citr(itr), nitr(itr);
			//sort right side
			for (++nitr; nitr != edges.end() && nitr->w < e.w; ++citr, ++nitr)
				*citr = *nitr;
			//sort left side
			if (citr == itr && citr != edges.begin())
			{
				nitr = itr;
				for (--nitr; nitr->w > e.w; --nitr, --citr)
				{
					*citr = *nitr;
					if (nitr == edges.begin())
					{
						--citr; break;
					}
				}
			}
			*citr = e;
		}
		return true;
	}

	//search the smallest region, and return its edge with smallest weight if its size is less than minRegionSize.
	_ItrT _searchMergeRegion(int minRegionSize, int RegionNode::* psize)
	{
		_ItrT mitr(edges.end());

		while (true)
		{
			int minSize = INT_MAX, mi = -1;
			for (int i = 0; i < (int)regions.size(); ++i)
			{
				int sz = regions[i].*psize;
				if (sz >= 0 && sz < minSize)
				{
					minSize = sz; mi = i;
				}
			}

			if (minSize > minRegionSize)
				break;

			float mw = FLT_MAX;

			for (_ItrT itr = edges.begin(); itr != edges.end(); ++itr)
			{
				if (itr->ri == mi || itr->rj == mi)
				{
					if (itr->w < mw)
					{
						mw = itr->w;
						mitr = itr;
					}
				}
			}

			if (mitr == edges.end())
				regions[mi].*psize = -1;
			else
				break;

		}

		return mitr;
	}
public:
	void build(const Im32SC1 &region, const Im32FC1 &edgeWeight, int minRegionSize, int minMaskSize, Im8UC1 sizeMask = Im8UC1())
	{
		this->_initSPGraph(region, edgeWeight, sizeMask);

		//merge small regions
		_ItrT eitr;
		if (minRegionSize > 0)
		{
			while ((eitr = this->_searchMergeRegion(minRegionSize, &RegionNode::size)) != edges.end())
				this->_removeEdge(eitr);
		}

		if (minMaskSize > 0)
		{
			while ((eitr = this->_searchMergeRegion(minMaskSize, &RegionNode::msize)) != edges.end())
				this->_removeEdge(eitr);
		}

		largeRegionStart = (int)edgesSorted.size();

		initRegion = region.clone();
	}
	int     nMerge() const
	{
		return (int)veq.size() - largeRegionStart;
	}
	Im32SC1 get(bool keepOriginalLabel = false)
	{
		if (edgesSorted.empty() || veq.empty())
			return initRegion;

		int i = largeRegionStart;

		std::unique_ptr<int[]> _remap(new int[initNCC]);
		int *remap = _remap.get();
		int ncc = cv::merge_eq_class((int(*)[2])&veq[0], i, initNCC, remap, keepOriginalLabel);
		Im32SC1 cc(initRegion.size());
		for_each_2(DWHN1(initRegion), DN1(cc), [remap](int s, int &d) {
			d = remap[s];
		});

		return cc;
	}
};


Mat1i segmentImage(const Mat1f &ucm, const std::vector<Match> &vmatch, int minRegionSize, int minRegionMatches, float ucmThreshold)
{
	Mat1i region;
	ucm2Region(ucm, ucmThreshold, region);

	Mat1b matchMask(ucm.size());
	matchMask = 0;
	for (auto &m : vmatch)
	{
		int x = int(m.x0 + 0.5f), y = int(m.y0 + 0.5f);
		if (uint(x)<uint(matchMask.cols) && uint(y)<uint(matchMask.rows) /*&& region(y,x)!=0*/)
			matchMask(y, x) = 255;
	}

	HSeg hs;
	hs.build(region, ucm, minRegionSize, minRegionMatches, matchMask);

	return hs.get(false);
}

