module NaiveBayes
  module Util
    def sum(arr)
      res = 0
      arr.each{|e| res+=e}
      res
    end

    def mult(arr)
      res = 1
      arr.each{|e| res*=e}
      res
    end
  end

  class Classifier
    include Util
    
    def initialize(training, laplace_factor=1.0)
      @laplace_factor = laplace_factor
      # dictionary per class, used to calculate conditional probability of words per class
      @dict = {}
      # dictionary over the whole training set. its size is used in laplace smoothing
      @global_dict = Hash.new(0.0)
      # total words per class
      @total_words = Hash.new(0.0)
      total_messages = 0
      training.each_value{|messages| total_messages += messages.size}
      @priors = {}
      
      # construct dictionaries and count words
      training.each_pair{|klass, messages|
        @dict[klass] = {}
        @dict[klass].default = 0.0
        messages.each{|msg|
          msg.split.each{|word|
          @dict[klass][word] += 1
            @global_dict[word] += 1
            @total_words[klass] += 1
          }
        }
        # calculate prio probability of this class
        @priors[klass] = (messages.size.to_f + @laplace_factor) / (total_messages.to_f + @laplace_factor * training.keys.size)
      }
      
      # store classes
      @classes = training.keys
    end

    # conditional probability of word, given klass
    def prob(word, klass)
      (@dict[klass][word] + @laplace_factor) / (@total_words[klass] + @laplace_factor * @global_dict.size)
    end

    def classify(message)
      res={}
      @classes.each{|klass|
        words = message.split
        num = mult(words.map{|w| prob(w, klass)}) * @priors[klass]
        denom = 0.0
        @classes.each{|kk|
          denom += mult(words.map{|w| prob(w, kk)}) * @priors[kk]
        }
        res[klass] = num/denom
      }
      res
    end
  end
end

if $0==__FILE__
  training = {
    :spam=>["offer is secret", "click secret link", "secret sports link"],
    :ham=>["play sports today", "went play sports", "secret sports event", "sports is today", "sports costs money"]
  }
  
  c = NaiveBayes::Classifier.new(training)
  
  p c
  puts c.prob("today", :spam)
  puts c.prob("today", :ham)
  
  p c.classify("today is secret")
end