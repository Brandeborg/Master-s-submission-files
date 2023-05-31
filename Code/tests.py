import data_preparation
from datasets import Dataset, DatasetDict, load_from_disk
import fine_tuning
import unittest
import os

from dotenv import load_dotenv
load_dotenv()  

# --- This file contains functions in charge of testing some of the more intricate functions of the project. There is not much, it is mostly to --- #
# --- confirm regular expressions do what they are meant to.                                                                                    --- #

# --- LOAD ENV CONSTANTS FOR CONSISTENT FILE NAMES --- #

class TestRemoveJunk(unittest.TestCase):
    def test_remove_junk_from_row(self):
        """Test if remove_junk_from_rows() removes the snippets it is supposed to, and doesn't remove what it should not.
        """
        dataset = Dataset.from_dict({"sentence": ["Just a regular sentence with a date at the end |10/12/2012 ",
                                                "Just a regular sentence with a date at the end and some extra|10.12.2012 - ABC.com ",
                                                "A sentence with a date in the middle 10.04.2020 of it | Aired 10-12-2012 | The Times Magazine ",
                                                "Some trash | And then something with a date 21.12.2022",
                                                "World News with David Muir : WN 09/16/15:  GOP Showdown Watch Full Episode | 09/16/2015 - ABC.com",
                                                "On This Day in History - January 2nd - Almanac - UPI.com| something happened",
                                                "Photos of the Day: July 27 - WSJ| We should really keep this part - WSJ",
                                                "Something with | junk on both sides",
                                                "Something with | a delcious middle but | junk on both sides",
                                                "a number with 10 digits, that is not a date | 0123456789",
                                                "July 23 (BusinessDesk) - oh yeah",
                                                "Photos of the Day: July 16 - something here",
                                                "A radical overhaul of Sydney&rsquo;s public bus transport system could cost commuters up to $1300 in additi...",
                                                "Reuters The European Commission has fined Credit Agricole <CAGR.PA>, HSBC <HSBA.L> and JPMorgan Chase <JPM.N> a total of 485 million euros",
                                                """Susan B. Glasser - POLITICO Magazine Susan Glasser is POLITICO’s chief international affairs columnist and host of its new weekly podcast, The Global Politico.<br/><br/>Glasser, who served as founding editor of the award-winning POLITICO Magazine and went on to become editor of POLITICO throughout the 2016 election cycle, has reported everywhere from the halls of Congress to the battle of Tora Bora. The former editor in chief of Foreign Policy magazine, she spent four years traveling the former Soviet Union as the Washington Post’s Moscow co-bureau chief, covered the wars in Iraq and Afghanistan and co-authored “Kremlin Rising: Vladimir Putin and the End of Revolution,” with her husband, New York Times chief White House correspondent Peter Baker. They’re now working on a biography of former Secretary of State Jim Baker. <br/><br/>Glasser joined POLITICO in 2013 after several years as editor-in-chief of the award-winning magazine Foreign Policy, overseeing its relaunch in print and as a daily online magazine. During her tenure, the magazine was recognized as a finalist for 10 National Magazine Awards and won three of the magazine world&#39;s highest honors. <br/><br/>Before that, Glasser worked for a decade at The Washington Post, where she was a foreign correspondent, editor of the Post’s Sunday Outlook and national news sections and political reporter. She started at the Post in 1998 as deputy national editor overseeing the Monica Lewinsky investigation and subsequent impeachment of President Bill Clinton, and later served as a national political reporter, covering the intersection of money and politics.<br/><br/>Prior to the Post, Glasser worked for eight years at Roll Call, the newspaper covering the U.S. Congress, where she rose from an intern to be the top editor. A graduate of Harvard University, Glasser lives in Washington with Baker and their son. She serves on the boards of the Pew Research Center and the Harvard Crimson student newspaper and is a visiting fellow at the Brookings Institution. <br/><div class="content layout-bi-unequal-fixed "><aside class="content-group module-enhanced-sign-up"> <section class="speedbump layout-vertical"> <div class="speedbump-item pos-alpha"> <div class="spotlight spotlight--flex"> <div class="summary link-alt fx1"> <a href="http://www.politico.com/tipsheets/the-global-politico/archive" target="_top"><h2>Sign up here for The Global Politico</h2></a><a href="http://www.politico.com/tipsheets/the-global-politico/archive" target="_top"> </a><p>Susan Glasser’s new weekly podcast takes you backstage in a world disrupted. Sign up here to receive new episodes in your inbox each Monday morning featuring exclusive, revealing conversations with leaders in Washington — and around the globe. </p> </div> </div> </div> <div class="speedbump-item pos-beta"> <div class="js-tealium-newsletter" data-subscription-module="newsletter_page_standard_The Global Politico - POLITICO"> <div class="dari-frame dari-frame-loaded" name="module-enhanced-sign-up-full-0000015b-ce49-dc47-a5df-fe4f9f370001" data-insertion-mode="replace" data-extra-form-data="_frame.path=5c388a2e37218b65f1db610df3d5c42a&amp;_frame.name=module-enhanced-sign-up-full-0000015b-ce49-dc47-a5df-fe4f9f370001"><form target="module-enhanced-sign-up-full-0000015b-ce49-dc47-a5df-fe4f9f370001" class="simple-signup simple-signup--stack" method="post" action="/subscribe/the-global-politico?"> <input type="hidden" name="subscribeId" value="0000015b-ce49-dc47-a5df-fe4f9f370001"/> <input type="hidden" name="processorId" value="0000015b-ce89-de92-a17b-ced9695b0000"/> <input type="hidden" name="validateEmail" value="true"/> <input type="hidden" name="enhancedSignUp" value="true"/> <input type="hidden" name="bot-field" value="" class="dn"/> <fieldset> <label for="i60d7adc675cd4b158cc619566c31cb87" class="simple-signup__label"> <span class="icon-text">Email</span> </label> <input type="email" name="subscribeEmail" id="i60d7adc675cd4b158cc619566c31cb87" value="" class="simple-signup__input " placeholder="Your email…"/> <button class="button type-link simple-signup__submit" type="submit" style="background: #b70000; border-radius: 25px;"> Sign Up</button> </fieldset> </form> <p class="legal-disclaimer">By signing up you agree to receive email newsletters or alerts from POLITICO. You can unsubscribe at any time.</p> </div></div> </div> </section> </aside></div>', 'Matt Friedman - POLITICO Matt Friedman is a reporter for POLITICO New Jersey and writes <a href="http://www.politico.com/newjerseyplaybook" target="_blank">New Jersey&#39;s Playbook</a>.<br/><br/>He has been reporting on New Jersey politics since 2007, beginning at PoliticsNJ.com (now PolitickerNJ.com) and followed by five and a half years at the Star-Ledger&#39;s Statehouse bureau. <br/><br/>Prior to reporting in New Jersey, Matt worked as a research assistant for Village Voice investigative reporter Wayne Barrett.<br/><br/><b>SUBSCRIBE </b>to POLITICO <b>New Jersey</b> <b>Playbook: </b><a href="https://www.politico.com/newjerseyplaybook" target="_blank">http://politi.co/1HLKltF</a><b> </b>"""]})
        
        cleaned_dataset = data_preparation.remove_junk_from_rows(dataset,
                                                                 columns=["sentence"],
                                                                 junk_snippets=["| The Times Magazine ", 
                                                                                "|12.12.2012", 
                                                                                "Some trash |", 
                                                                                "Something with |", 
                                                                                "| junk on both sides",
                                                                                "- WSJ"])

        expected = ['Just a regular sentence with a date at the end', 
                    'Just a regular sentence with a date at the end and some extra', 
                    'A sentence with a date in the middle 10.04.2020 of it', 
                    'And then something with a date 21.12.2022',
                    "GOP Showdown Watch Full Episode",
                    "something happened",
                    "| We should really keep this part",
                    "junk on both sides",
                    "a delcious middle but",
                    "a number with 10 digits, that is not a date | 0123456789",
                    "oh yeah",
                    "- something here",
                    "A radical overhaul of Sydney\u2019s public bus transport system could cost commuters up to $1300 in ...",
                    "Reuters The European Commission has fined Credit Agricole <CAGR.PA>, HSBC <HSBA.L> and JPMorgan Chase <JPM.N> a total of 485 million euros",
                    """Susan B. Glasser - POLITICO Magazine Susan Glasser is POLITICO’s chief international affairs columnist and host of its new weekly podcast, The Global Politico.Glasser, who served as founding editor of the award-winning POLITICO Magazine and went on to become editor of POLITICO throughout the 2016 election cycle, has reported everywhere from the halls of Congress to the battle of Tora Bora. The former editor in chief of Foreign Policy magazine, she spent four years traveling the former Soviet Union as the Washington Post’s Moscow co-bureau chief, covered the wars in Iraq and Afghanistan and co-authored “Kremlin Rising: Vladimir Putin and the End of Revolution,” with her husband, New York Times chief White House correspondent Peter Baker. They’re now working on a biography of former Secretary of State Jim Baker. Glasser joined POLITICO in 2013 after several years as editor-in-chief of the award-winning magazine Foreign Policy, overseeing its relaunch in print and as a daily online magazine. During her tenure, the magazine was recognized as a finalist for 10 National Magazine Awards and won three of the magazine world's highest honors. Before that, Glasser worked for a decade at The Washington Post, where she was a foreign correspondent, editor of the Post’s Sunday Outlook and national news sections and political reporter. She started at the Post in 1998 as deputy national editor overseeing the Monica Lewinsky investigation and subsequent impeachment of President Bill Clinton, and later served as a national political reporter, covering the intersection of money and politics.Prior to the Post, Glasser worked for eight years at Roll Call, the newspaper covering the U.S. Congress, where she rose from an intern to be the top editor. A graduate of Harvard University, Glasser lives in Washington with Baker and their son. She serves on the boards of the Pew Research Center and the Harvard Crimson student newspaper and is a visiting fellow at the Brookings Institution. ', 'Matt Friedman - POLITICO Matt Friedman is a reporter for POLITICO New Jersey and writes New Jersey's Playbook.He has been reporting on New Jersey politics since 2007, beginning at PoliticsNJ.com (now PolitickerNJ.com) and followed by five and a half years at the Star-Ledger's Statehouse bureau. Prior to reporting in New Jersey, Matt worked as a research assistant for Village Voice investigative reporter Wayne Barrett.SUBSCRIBE to POLITICO New Jersey Playbook: http://politi.co/1HLKltF"""]
    
        actual = cleaned_dataset[:]["sentence"]
        
        self.assertEqual(actual, expected)

class TestFilterJunk(unittest.TestCase):
    def test_remove_whole_junk_row(self):
        """Test if remove_whole_junk_rows() removes the right rows and does not remove what it should not.
        """
        dataset = Dataset.from_dict({"title":["| a href=\"http://www.cbsnews.com/news/48-hours-hannah-graham-deadly-connections/\" target=\"new\"><b>Read story</b></a>", 
                                              "A nation divided - WND Cast your vote now. All answers are stored anonymously. ADD THIS POLL TO YOUR SITE (copy the code below) <iframe src=\"http://www.wnd.com/man-in-charge/\" style=\"width: 600px; height: 582px; border: 1px;\"></iframe>",
                                              "<p>This should not be removed</p>"], "description": ["","",""]})
        filtered_dataset = data_preparation.remove_whole_junk_rows(dataset)

        self.assertEqual(filtered_dataset[:]["title"], ["<p>This should not be removed</p>"])
    
class TestPBETokenizer(unittest.TestCase):
    def test_PBE_tokenizer_by_word(self):
        """Tests if tokenizing word by word has the same result as tokenizing everything at once.
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

        def BPE_tokenize_by_word(words: str, tokenizer):
            lists_of_tokens = [tokenizer.tokenize(word) for word in words]
            tokens = [token for sublist in lists_of_tokens for token in sublist]

            return tokens

        test_sent = "This is a sentence with named entities, yeah. One of them is Biff Buff. He lives in Jylland with his son Buffo."
        test_sent_messed_up = " This is a sentence with named entities, yeah. One of!' them is Biff ¤%& Buff. He lives, asobb2  .S.BSÆOIFBæw hæOWFBwøpweifb wP9FVWEF in Jylland with his son Buffo.2938y4."
        
        self.assertEqual(BPE_tokenize_by_word(test_sent_messed_up.split(" "), tokenizer), tokenizer.tokenize(test_sent_messed_up.split(" "), is_split_into_words=True))

class TestRemoveDuplicates(unittest.TestCase):
    def test_remove_duplicates(self):
        """Test remove_duplicates() works.

        Yields:
            _type_: _description_
        """
        test = {'meta':        [{'article_id': 'bb3fbc550c9368d559da19f08352febd', 'date': '20150717', 'outlet': 'afr.com'}, 
                                {'article_id': 'bb3fbc550c9368d559da19f08352febd', 'date': '20150717', 'outlet': 'afr.com'}, 
                                {'article_id': '7d144bc1fb9e985c84ef4533d9696454', 'date': '20150717', 'outlet': 'afr.com'}, 
                                {'article_id': '24712f3bf7b85d6c9afb8c0760eab0bc', 'date': '20150717', 'outlet': 'afr.com'}], 
                'title':       ["Behind the scenes with Hollywood's pretend-penis maker", 
                                "Behind the scenes with Hollywood's pretend-penis maker", 
                                "ALE Property values rise as average yield drops under 6 per cent", 
                                "IT firm alleges Domino's Pizza stole its GPS tracking technology"], 
                'description': ["Matthew Mungle's business is on the rise: he's the go-to man when filmmakers need a prosthetic penis. ",
                                "Matthew Mungle's business is on the rise: he's the go-to man when filmmakers need a prosthetic penis. ", 
                                "ALE Property Group has reported a valuation boost to its now $900.5 million property portfolio, driven by a tightening in the average capitalisation rate to 5.99 per cent. ", 
                                "A Sydney IT firm has claimed Domino's Pizza Enterprises stole the technology used in its GPS tracking system that allows customers to follow their pizza from store to door. "]}
        test_dataset = Dataset.from_dict(test)
        no_dups = data_preparation.remove_duplicates(test_dataset)

        self.assertEqual(1, test_dataset.num_rows - no_dups.num_rows)

def main():
    test_remove_junk = TestRemoveJunk()
    test_remove_junk.test_remove_junk_from_row()

    test_filter_junk = TestFilterJunk()
    test_filter_junk.test_remove_whole_junk_row()

    test_BPE_tokenizer = TestPBETokenizer()
    test_BPE_tokenizer.test_PBE_tokenizer_by_word()

    test_remove_duplicates = TestRemoveDuplicates()
    test_remove_duplicates.test_remove_duplicates()

if __name__ == "__main__":
    main()
