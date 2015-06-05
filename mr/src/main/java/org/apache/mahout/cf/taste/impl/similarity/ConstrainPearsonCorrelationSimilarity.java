package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;

public final class ConstrainPearsonCorrelationSimilarity extends AbstractSimilarity {

	public ConstrainPearsonCorrelationSimilarity(DataModel dataModel) throws TasteException {
		this(dataModel, Weighting.UNWEIGHTED);
	}
	
	public ConstrainPearsonCorrelationSimilarity(DataModel dataModel,
			Weighting weighting) throws TasteException {
		super(dataModel, weighting, false);
	}

	@Override
	double computeResult(int n, double sumXY, double sumX2, double sumY2,
			double sumXYdiff2) {
		if (n == 0) {
	      return Double.NaN;
	    }
		DataModel dataModel = getDataModel();
		double sumX = Math.sqrt(sumX2);
		double sumY = Math.sqrt(sumY2);
		float maxPref = dataModel.getMaxPreference();
		float minPref = dataModel.getMinPreference();
		//median value
		double median = (minPref + maxPref) / 2;
		double denominator = Math.sqrt(sumX2 + n * Math.pow(median, 2) - 2 * sumX * median) * 
						     Math.sqrt(sumY2 + n * Math.pow(median, 2) - 2 * sumY * median);
		if (denominator == 0.0) {
	      return Double.NaN;
	    }
		return (sumXY - (sumX + sumY) * median + n * Math.pow(median, 2)) / denominator;
	}
}
