package SearchLucene;

import Classes.Document;
import Classes.Query;
import IndexingLucene.MyIndexReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Borrow my code from hw3
 */
public class HW3QRModel {

    private final MyIndexReader indexReader;
    private final long collectionTotalLength;
    private double mu = 2000;
    private HashMap<String, Long> collectionFreq = new HashMap<>();
    private HashMap<String, int[][]> collectionPostings = new HashMap<>();

    public HW3QRModel(MyIndexReader ixreader) {
        indexReader = ixreader;
        this.collectionTotalLength = ixreader.getTotalContentLength();
    }

    public double getMu() {
        return mu;
    }

    public void setMu(double mu) {
        this.mu = mu;
    }

    /**
     * Search for the topic information.
     * The returned results (retrieved documents) should be ranked by the score (from the most relevant to the least).
     * TopN specifies the maximum number of results to be returned.
     * NT: you will find our IndexingLucene.Myindexreader provides method: docLength()
     * implement your retrieval model here, and for each input query, return the topN retrieved documents
     * sort the documents based on their relevance score, from high to low
     *
     * @param aQuery The query to be searched for.
     * @param TopN   The maximum number of returned document
     */
    public List<Document> retrieveQuery(Query aQuery, int TopN) throws IOException {
        String[] queryTokens = aQuery.GetQueryContent().split(" ");
        if (queryTokens.length == 0) return new ArrayList<>(0);

        return internalQueryDocumentRanked(queryTokens, TopN);
    }

    /**
     * Internal method for querying one tokenized document, rank the result based on scores
     */
    private List<Document> internalQueryDocumentRanked(String[] tokens, int topN) throws IOException {
        HashMap<Integer, HashMap<String, Integer>> queryResult = populateQueryResult(tokens);
        ArrayList<Document> allResults = queryLikelihood(queryResult, tokens);

        // Order by score DESC
        allResults.sort((doc1, doc2) -> {
            // (d1 <= d2)
            double d1s = doc1.score(), d2s = doc2.score();
            return d1s > d2s ? -1 : 1;
        });

        // Pick top N results
        int finalSize = Math.min(topN, allResults.size());
        ArrayList<Document> res = new ArrayList<>(finalSize);
        int countDoc = 0;
        for (Document doc : allResults) {
            res.add(doc);
            if (++countDoc > finalSize - 1) break;
        }
        // Save memory
        queryResult.forEach((id, tmp) -> tmp.clear());
        queryResult.clear();
        allResults.clear();
        return res;
    }

    private HashMap<Integer, HashMap<String, Integer>> populateQueryResult(String[] tokens) throws IOException {
        // <DOCID, <TERM, FREQ>>
        HashMap<Integer, HashMap<String, Integer>> tokenOnCollection = new HashMap<>();
        for (String token : tokens) {
            Long cFreq = getCollectionFreq(token);
            // Non-exist, no need to calc posting list
            if (cFreq.equals(0L)) continue;
            int[][] postingList = getCollectionPostings(token);
            for (int[] postingForOneDoc : postingList) {
                int docid = postingForOneDoc[0], docFreq = postingForOneDoc[1];
                HashMap<String, Integer> oneTermFreq = tokenOnCollection.getOrDefault(docid, new HashMap<>());
                if (oneTermFreq.size() == 0) {
                    tokenOnCollection.putIfAbsent(docid, oneTermFreq);
                }
                oneTermFreq.put(token, docFreq);
            }
        }
        return tokenOnCollection;
    }

    /**
     * Cache collection posting list and get the cached result
     */
    private int[][] getCollectionPostings(String token) throws IOException {
        if (!this.collectionPostings.containsKey(token)) {
            int[][] postingList = this.indexReader.getPostingList(token);
            if (postingList == null) postingList = new int[0][];
            // Show a warning about detecting non-exist term token
            if (postingList.length == 0)
                System.err.println(String.format("[WARN] Token <%s> not in collection", token));
            this.collectionPostings.put(token, postingList);
        }
        return this.collectionPostings.get(token);
    }

    private Long calcCollectionFreq(int[][] postings) {
        long count = 0;
        for (int[] one : postings) {
            count += one[1];
        }
        return count;
    }

    /**
     * Get term freq in the given collection of given token
     */
    private Long getCollectionFreq(String token) throws IOException {
        if (!this.collectionFreq.containsKey(token)) {
            Long termFreq = this.indexReader.CollectionFreq(token);
            Long myFreq = calcCollectionFreq(getCollectionPostings(token));
            if (!myFreq.equals(termFreq))
                System.err.println("Collection frequency disagree.");
            this.collectionFreq.put(token, termFreq);
        }
        return this.collectionFreq.get(token);
    }

    /**
     * Use LM method for calc the query result score
     */
    private ArrayList<Document> queryLikelihood(HashMap<Integer, HashMap<String, Integer>> queryResult, String[] tokens)
            throws IOException {
        ArrayList<Document> allResults = new ArrayList<>(queryResult.size());
        for (Integer docid : queryResult.keySet()) {
            double score = getScore(tokens, queryResult.get(docid), this.indexReader.docLength(docid));
            Document d = new Document(docid.toString(), this.indexReader.getDocno(docid), score);
            allResults.add(d);
        }
        return allResults;
    }

    /**
     * Dirichlet smoothing (Reference: org.apache.lucene.search.similarities.LMDirichletSimilarity)
     */
    private double getScore(String[] tokens, HashMap<String, Integer> docTermFreqList, int doclen) {
        double score = 1.0;
        double adjLen = (doclen + this.mu);
        double // (|D|/(|D|+MU)) as l1 and (MU/(|D|+MU)) as r1
                l1 = 1.0 * doclen / adjLen,
                r1 = 1.0 * this.mu / adjLen;
        for (String token : tokens) {
            Long cf = this.collectionFreq.get(token);
            // Non-exist, no need to calc rest
            if (cf.equals(0L)) continue;
            int tf = docTermFreqList.getOrDefault(token, 0);
            //System.err.println(String.format("Doc freq: %d,%d", tf, tf1));
            double // p(w|D) = l1*(c(w,D)/|D|) + r1*p(w|REF)
                    l2 = 1.0 * tf / doclen,
                    r2 = 1.0 * cf / this.collectionTotalLength;
            // Unigram LM
            score *= (l1 * l2 + r1 * r2);
        }
        score = score > 0 ? score : 0;
        return score;
    }

}
