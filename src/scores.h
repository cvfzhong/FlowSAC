#pragma once

#include"flowsac.h"

//void calcSmoothCost(const Mat2f &flow, const Mat1i &seg, int nRegions, std::vector<double> &cost, int d)
//{
//	cost.clear();
//	cost.resize(nRegions, 0);
//
//	std::vector<double> vn(nRegions, 0);
//
//	auto addCost = [&cost, &vn](const Vec2f &p, int pr, const Vec2f &q, int qr) {
//		if (pr == qr)
//		{
//			Vec2f d = p - q;
//			double v = sqrt(d.dot(d));
//			cost[pr] += v;
//			cost[qr] += v;
//			vn[pr] += 1;
//			vn[qr] += 1;
//		}
//	};
//
//	for (int y = 0; y < flow.rows - d; ++y)
//	{
//		for (int x = 0; x < flow.cols - d; ++x)
//		{
//			Vec2f p = flow(y, x) - Vec2f(x, y);
//			int pr = seg(y, x);
//
//			addCost(p, pr, flow(y, x + d) - Vec2f(x + d, y), seg(y, x + d));
//			addCost(p, pr, flow(y + d, x + d) - Vec2f(x + d, y + d), seg(y + d, x + d));
//			addCost(p, pr, flow(y + d, x) - Vec2f(x, y + d), seg(y + d, x));
//			if (x - d >= 0)
//				addCost(p, pr, flow(y + d, x - d) - Vec2f(x - d, y + d), seg(y + d, x - d));
//		}
//	}
//	for (int i = 0; i < nRegions; ++i)
//	{
//		cost[i] /= vn[i] + 1e-8;
//	}
//}

class SOD
	:public FlowScore
{
	Mat1i _seg;
	int   _nRegions;

	Mat3b _src, _tar;
	Mat1f _srcf, _tarf;
public:
	static Mat1f getDirs(const cv::Mat3b &img, double smoothSigma)
	{
		Mat3f imgf;
		img.convertTo(imgf, CV_32F, 1.0 / 255);

		Mat1f gray;
		cvtColor(imgf, gray, CV_BGR2GRAY);

		if (smoothSigma>1e-3)
			GaussianBlur(gray, gray, Size(5, 5), smoothSigma);

		const int ksize = 3;
		Mat1f dx, dy;
		cv::Sobel(gray, dx, CV_32F, 1, 0, ksize);
		cv::Sobel(gray, dy, CV_32F, 0, 1, ksize);

		Mat1f dir(gray.size());
		for_each_3(DWHN1(dx), DN1(dy), DNC(dir), [](float dx, float dy, float *v) {
			*v = atan2(dy, dx);
		});

		return dir;
	}
	SOD(const Mat3b &src, const Mat3b &tar, const Mat1i &seg, double smoothSigma)
		:_seg(seg), _src(src), _tar(tar), _srcf(getDirs(src, smoothSigma)), _tarf(getDirs(tar, smoothSigma))
	{
		_nRegions = maxElem(seg) + 1;
	}
	void exec(const Mat2f &flow, std::vector<double> &regionScores)
	{
		regionScores.resize(_nRegions);
		double *vs = &regionScores[0];
		memset(vs, 0, sizeof(double)*_nRegions);

		std::vector<double>  sumn(_nRegions, 0.0);

		const float dT = CV_PI / 3;
		Mat1f tarf(_tarf);

		for_each_3(DWHN1(_seg), DN1(_srcf), DNC(flow), [&tarf, vs, &sumn, dT](int r, const float p, const float *f) {
			int tx = int(f[0] + 0.5), ty = int(f[1] + 0.5);
			if (/*m!=0 &&*/ uint(tx) < tarf.cols && uint(ty) < tarf.rows)
			{
				float d = tarf(ty, tx) - p;
				if (d < 0)
					d = -d;

				if (d > CV_PI)
					d = 2 * CV_PI - d;

				d = d > dT ? dT : d;

				vs[r] += dT - d;
				sumn[r] += 1;
			}
		});
		for (size_t i = 0; i < sumn.size(); ++i)
			regionScores[i] /= sumn[i] + 1e-8;

		//if (_alpha > 0)
		//{
		//	std::vector<double> smoothCost;
		//	calcSmoothCost(flow, _seg, _nRegions, smoothCost, _d);
		//	for (size_t i = 0; i < regionScores.size(); ++i)
		//		regionScores[i] -= _alpha*smoothCost[i];
		//}
	}
};



class SAD
	:public FlowScore
{
	Mat1i _seg;
	Mat3f _src, _tar;
	int   _nRegions;
public:
	static Mat3f cvt(const Mat3b &img)
	{
		Mat3f imgf;
		img.convertTo(imgf, CV_32F, 1.0f / 255);
		return imgf;
	}
	SAD(const Mat3b &src, const Mat3b &tar, const Mat1i &seg)
		:_seg(seg), _src(cvt(src)), _tar(cvt(tar))
	{
		_nRegions = maxElem(seg) + 1;
	}
	void exec(const Mat2f &flow, std::vector<double> &regionScores)
	{
		regionScores.resize(_nRegions);
		double *vs = &regionScores[0];
		memset(vs, 0, sizeof(double)*_nRegions);

		std::vector<double>  sumn(_nRegions, 0.0);

		Mat3f tar(_tar);
		for_each_3(DWHN1(_seg), DNC(_src), DNC(flow), [&tar, vs, &sumn](int r, const float *p, const float *f) {
			int tx = int(f[0] + 0.5), ty = int(f[1] + 0.5);
			if (uint(tx) < tar.cols && uint(ty) < tar.rows)
			{
				const float *q = tar.ptr<float>(ty, tx);
				double d = 0;
				for (int i = 0; i < 3; ++i)
					d += 1.0 - abs(p[i] - q[i]);
				vs[r] += d;
				sumn[r] += 1;
			}
		});
		for (size_t i = 0; i < sumn.size(); ++i)
			regionScores[i] /= sumn[i] + 1e-8;
	}
};



class SNCC
	:public FlowScore
{
	Mat1i _seg;
	int   _nRegions;
	int   _wsz;
	Mat3f _src, _tar;

	Mat3f srcf, tarf;
	Mat3f srcMean, tarMean;
public:
	void _setImage(const Mat3b &img, Mat3f &_img, Mat3f &imgf, Mat3f &mean, int wsz)
	{
		Mat imgx;
		copyMakeBorder(img, imgx, wsz / 2, wsz / 2, wsz / 2, wsz / 2, BORDER_REFLECT);
		imgx.convertTo(_img, CV_32F, 1.0 / 255);

		Rect roi(wsz / 2, wsz / 2, img.cols, img.rows);
		imgf = _img(roi);

		Mat3f meanx(_img.size());
		boxFilter(_img, meanx, CV_32F, Size(wsz, wsz));
		mean = meanx(roi).clone();
	}
	SNCC(const Mat3b &src, const Mat3b &tar, const Mat1i &seg, int wsz)
		:_seg(seg), _wsz(wsz)
	{
		_nRegions = maxElem(seg) + 1;
		_setImage(src, _src, srcf, srcMean, wsz);
		_setImage(tar, _tar, tarf, tarMean, wsz);
	}
	static float _calcNCC(const float *src, const int sstride, float *srcMean, const float *tar, const int tstride, float *tarMean, const int wsz, const int ds)
	{
		float r = 0;

		const int hwsz = wsz / 2;
		src -= sstride*hwsz + hwsz * 3;
		tar -= tstride*hwsz + hwsz * 3;

		float slen = 0, tlen = 0, c = 0;
		//float sm = (srcMean[0] + srcMean[1] + srcMean[2]) / 3, tm = (tarMean[0] + tarMean[1] + tarMean[2]) / 3;

		for (int y = 0; y < wsz; y += ds, src += sstride*ds, tar += tstride*ds)
		{
			const float *s = src, *t = tar;
			for (int x = 0; x < wsz; x += ds, s += ds * 3, t += ds * 3)
			{
				for (int j = 0; j < 3; ++j)
				{
					float a = s[j] - srcMean[j], b = t[j] - tarMean[j];
					//float a = s[j] - sm, b = t[j] - tm;
					slen += a*a;
					tlen += b*b;
					c += a*b;
				}
			}
		}

		c = c / (sqrt(slen*tlen) + 1e-12);
		return c;
	}
	
	void exec(const Mat2f &flow, std::vector<double> &regionScores)
	{
		regionScores.resize(_nRegions);
		memset(&regionScores[0], 0, sizeof(double)*_nRegions);

		std::vector<double>  sumn(_nRegions, 0.0);

		for (int y = 0; y<srcf.rows; ++y)
			for (int x = 0; x < srcf.cols; ++x)
			{
				const float *f = flow.ptr<float>(y, x);
				int tx = int(f[0] + 0.5), ty = int(f[1] + 0.5);
				if (uint(tx) < tarf.cols && uint(ty) < tarf.rows)
				{
					float v = _calcNCC(srcf.ptr<float>(y, x), stepC(srcf), srcMean.ptr<float>(y, x), tarf.ptr<float>(ty, tx), stepC(tarf), tarMean.ptr<float>(ty, tx), _wsz, 1);
					if (v < 0)
						v = 0;
					int ri = _seg(y, x);
					regionScores[ri] += v;
					sumn[ri] += 1;
				}
			}

		for (size_t i = 0; i < sumn.size(); ++i)
			regionScores[i] /= sumn[i] + 1e-8;
	}
};

class SCT
	:public FlowScore
{
	Mat1i _seg;
	int   _nRegions;
	int   _wsz;

	Mat3f _src, _tar;
	Mat3f srcf, tarf;
public:
	void _setImage(const Mat3b &img, Mat3f &_img, Mat3f &imgf, int wsz)
	{
		Mat imgx;
		copyMakeBorder(img, imgx, wsz / 2, wsz / 2, wsz / 2, wsz / 2, BORDER_REFLECT);
		imgx.convertTo(_img, CV_32F, 1.0 / 255);

		Rect roi(wsz / 2, wsz / 2, img.cols, img.rows);
		imgf = _img(roi);
	}
	SCT(const Mat3b &src, const Mat3b &tar, const Mat1i &seg, int wsz)
		:_seg(seg), _wsz(wsz)
	{
		_nRegions = maxElem(seg) + 1;
		_setImage(src, _src, srcf, wsz);
		_setImage(tar, _tar, tarf, wsz);
	}
	static float _calcCensus(const float *src, const int sstride, const float *tar, const int tstride, const int wsz, const int ds)
	{
		const float *sc = src, *tc = tar;

		const int hwsz = wsz / 2;
		src -= sstride*hwsz + hwsz * 3;
		tar -= tstride*hwsz + hwsz * 3;

		float r = 0;

		for (int y = 0; y < wsz; y += ds, src += sstride*ds, tar += tstride*ds)
		{
			const float *s = src, *t = tar;
			for (int x = 0; x < wsz; x += ds, s += ds * 3, t += ds * 3)
			{
				for (int j = 0; j < 3; ++j)
				{
					if (s[j] > sc[j] && t[j] > tc[j] || s[j] < sc[j] && t[j] < tc[j])
						r += 1;
				}
			}
		}

		return r;
	}
	void exec(const Mat2f &flow, std::vector<double> &regionScores)
	{
		regionScores.resize(_nRegions);
		memset(&regionScores[0], 0, sizeof(double)*_nRegions);

		std::vector<double>  sumn(_nRegions, 0.0);

		for (int y = 0; y<srcf.rows; ++y)
			for (int x = 0; x < srcf.cols; ++x)
			{
				const float *f = flow.ptr<float>(y, x);
				int tx = int(f[0] + 0.5), ty = int(f[1] + 0.5);
				if (uint(tx) < tarf.cols && uint(ty) < tarf.rows)
				{
					float v = _calcCensus(srcf.ptr<float>(y, x), stepC(srcf), tarf.ptr<float>(ty, tx), stepC(tarf), _wsz, 1);

					int ri = _seg(y, x);
					regionScores[ri] += v;
					sumn[ri] += 1;
				}
			}

		for (size_t i = 0; i < sumn.size(); ++i)
			regionScores[i] /= sumn[i] + 1e-8;
	}
};

