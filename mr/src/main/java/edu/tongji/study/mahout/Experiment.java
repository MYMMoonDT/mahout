package edu.tongji.study.mahout;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.AdjustedCosineSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.ConstrainPearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.ImprovedNHSMSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.JMSDSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.JPSSSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.JaccardSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.MSDSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.NHSMSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PIPSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PSSSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.SigmoidPearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.UncenteredCosineSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.WeightedPearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

public class Experiment {
	public static void main(String[] args) throws IOException, TasteException{
		//RandomUtils.useTestSeed();
		
		DataModel model = new FileDataModel(new File("data/u.data"));

		//平均绝对误差
		RecommenderEvaluator evalAAD = new AverageAbsoluteDifferenceRecommenderEvaluator();
		//均方根误差
		RecommenderEvaluator evalRMS = new RMSRecommenderEvaluator();
		//准确率和召回率
		RecommenderIRStatsEvaluator evalIRStats = new GenericRecommenderIRStatsEvaluator(); 

		RecommenderBuilder PearsonCorrelationBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new PearsonCorrelationSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder UncenteredCosineBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new UncenteredCosineSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder ConstrainPearsonCorrelationBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new ConstrainPearsonCorrelationSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder WeightedPearsonCorrelationBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new WeightedPearsonCorrelationSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder SigmoidPearsonCorrelationBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new SigmoidPearsonCorrelationSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder AdjustedCosineBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new AdjustedCosineSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder JaccardBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new JaccardSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder MSDBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new MSDSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder JMSDBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new JMSDSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder PIPBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new PIPSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder PSSBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new PSSSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder JPSSBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new JPSSSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder NHSMBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new NHSMSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		RecommenderBuilder ImprovedNHSMBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)
					throws TasteException {
				UserSimilarity similarity = new ImprovedNHSMSimilarity(
						model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		
		/*
		System.out.println("----- PearsonCorrelation -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(PearsonCorrelationBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(PearsonCorrelationBuilder, null, model, 0.8, 1.0));
		IRStatistics stats = evalIRStats.evaluate(PearsonCorrelationBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- PearsonCorrelation -----");
		
		System.out.println("\n");
		
		System.out.println("----- UncenteredCosine -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(UncenteredCosineBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(UncenteredCosineBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(UncenteredCosineBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- UncenteredCosine -----");
		
		System.out.println("\n");
		
		System.out.println("----- ConstrainPearsonCorrelation -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(ConstrainPearsonCorrelationBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(ConstrainPearsonCorrelationBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(ConstrainPearsonCorrelationBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- ConstrainPearsonCorrelation -----");
		
		System.out.println("\n");
		
		System.out.println("----- WeightedPearsonCorrelation -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(WeightedPearsonCorrelationBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(WeightedPearsonCorrelationBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(WeightedPearsonCorrelationBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- WeightedPearsonCorrelation -----");
		
		System.out.println("\n");
		
		System.out.println("----- SigmoidPearsonCorrelation -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(SigmoidPearsonCorrelationBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(SigmoidPearsonCorrelationBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(SigmoidPearsonCorrelationBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- SigmoidPearsonCorrelation -----");
		
		System.out.println("\n");
		
		System.out.println("----- AdjustedCosine -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(AdjustedCosineBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(AdjustedCosineBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(AdjustedCosineBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- AdjustedCosine -----");
		
		System.out.println("\n");
		
		System.out.println("----- Jaccard -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(JaccardBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(JaccardBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(JaccardBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- Jaccard -----");
		
		System.out.println("\n");
		
		System.out.println("----- MSD -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(MSDBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(MSDBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(MSDBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- MSD -----");
		
		System.out.println("\n");
		
		System.out.println("----- JMSD -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(JMSDBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(JMSDBuilder, null, model, 0.8, 1.0));
		IRStatistics stats = evalIRStats.evaluate(JMSDBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- JMSD -----");
		
		System.out.println("\n");
		
		
		System.out.println("----- PIP -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(PIPBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(PIPBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(PIPBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- PIP -----");
		
		System.out.println("\n");
		
		System.out.println("----- PSS -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(PSSBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(PSSBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(PSSBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- PSS -----");
		
		System.out.println("\n");
		
		
		System.out.println("----- JPSS -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(JPSSBuilder, null, model, 0.8, 1.0));
		System.out.println("均方根误差: " + evalRMS.evaluate(JPSSBuilder, null, model, 0.8, 1.0));
		stats = evalIRStats.evaluate(JPSSBuilder, null, model, null, 100, 3, 1.0);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- JPSS -----");
		
		System.out.println("\n");
		*/
		
		System.out.println("----- NHSM -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(NHSMBuilder, null, model, 0.8, 0.2));
		System.out.println("均方根误差: " + evalRMS.evaluate(NHSMBuilder, null, model, 0.8, 0.2));
		IRStatistics stats = evalIRStats.evaluate(NHSMBuilder, null, model, null, 100, 3, 0.2);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- NHSM -----");
		
		System.out.println("\n");
		
		System.out.println("----- ImprovedNHSM -----");
		System.out.println("平均绝对误差: " + evalAAD.evaluate(ImprovedNHSMBuilder, null, model, 0.8, 0.2));
		System.out.println("均方根误差: " + evalRMS.evaluate(ImprovedNHSMBuilder, null, model, 0.8, 0.2));
		stats = evalIRStats.evaluate(ImprovedNHSMBuilder, null, model, null, 100, 3, 0.2);
		System.out.println("准确率: " + stats.getPrecision());
		System.out.println("召回率: " + stats.getRecall());
		System.out.println("----- ImprovedNHSM -----");
		
		System.out.println("\n");
	}
}
