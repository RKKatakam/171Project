import React, { useState, useEffect } from 'react';
import StarRate from "../scripts/stars";
import "../styles/Survey.css";

function Survey() {
    // Updated movie titles to match your dataset better
    const titles = [
        "10 Things I Hate About You (1999)", 
        "Alien (1979)", // Use the original Alien instead of Aliens
        "Nightmare Before Christmas, The (1993)", // Correct title format
        "Bohemian Rhapsody (2018)", 
        "Conjuring, The (2013)", // Correct title format
        "Star Wars: Episode V - The Empire Strikes Back (1980)", // Full title
        "Everything Everywhere All at Once (2022)", 
        "Guardians of the Galaxy Vol. 3 (2023)", 
        "Halloween (1978)", // Use original Halloween
        "Happy Gilmore (1996)", 
        "How to Train Your Dragon (2010)", 
        "Hunger Games, The (2012)", // Correct title format
        "Lilo & Stitch (2002)", 
        "Mission: Impossible III (2006)", // Note the Roman numerals
        "Godzilla Minus One (2023)", 
        "Fantastic Mr. Fox (2009)", 
        "Ocean's Eleven (2001)", // Use the remake, not the 1960 version
        "A Man Called Otto (2022)", 
        "Devil Wears Prada, The (2006)", // Correct title format
        "Shutter Island (2010)", 
        "Sicario (2015)", 
        "Theory of Everything, The (2014)", // Correct title format
        "Toy Story 3 (2010)", 
        "Twilight (2008)" // Use the vampire movie, not the 1998 one
    ];

    // now that I have the API implemented, I could rework this with said API, buut I'm lowk too lazy
    const images = [
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/ujERk3aKABXU3NDXOAxEQYTHe9A.jpg", // 10 Things I hate About You
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/vfrQk5IPloGg1v9Rzbh2Eg3VGyM.jpg", // Alien
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/oQffRNjK8e19rF7xVYEN8ew0j7b.jpg", // The Nightmare Before Christmas
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/lHu1wtNaczFPGFDTrjCSzeLPTKN.jpg", // Bohemian Rhapsody
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/wVYREutTvI2tmxr6ujrHT704wGF.jpg", // The Conjuring
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/nNAeTmF4CtdSgMDplXTDPOpYzsX.jpg", // The Empire Strikes Back
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/u68AjlvlutfEIcpmbYpKcdi09ut.jpg", // Everything Everywhere All at Once
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/r2J02Z2OpNTctfOSN1Ydgii51I3.jpg", // Guardians of the Galaxy: Volume 3
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/wijlZ3HaYMvlDTPqJoTCWKFkCPU.jpg", // Halloween
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/4RnCeRzvI1xk5tuNWjpDKzSnJDk.jpg", // Happy Gilmore
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/ygGmAO60t8GyqUo9xYeYxSZAR3b.jpg", // How to Train Your Dragon
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/yXCbOiVDCxO71zI7cuwBRXdftq8.jpg", // The Hunger Games
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/m13Vbzv7R2GMAl3GXFrkmMEgCFQ.jpg", // Lilo & Stitch
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/vKGYCpmQyV9uHybWDzXuII8Los5.jpg", // Mission: Impossible 3
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/2E2WTX0TJEflAged6kzErwqX1kt.jpg", // Godzilla Minus One
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/njbTizADSZg4PqeyJdDzZGooikv.jpg", // Fantastic Mr. Fox
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/hQQCdZrsHtZyR6NbKH2YyCqd2fR.jpg", // Ocean's Eleven
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/130H1gap9lFfiTF9iDrqNIkFvC9.jpg", // A Man Called Otto
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/8912AsVuS7Sj915apArUFbv6F9L.jpg", // The Devil Wears Prada
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/nrmXQ0zcZUL8jFLrakWc90IR8z9.jpg", // Shutter Island
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/lz8vNyXeidqqOdJW9ZjnDAMb5Vr.jpg", // Sicario
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/kJuL37NTE51zVP3eG5aGMyKAIlh.jpg", // The Theory of Everything
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/AbbXspMOwdvwWZgVN0nabZq03Ec.jpg", // Toy Story 3
        "https://media.themoviedb.org/t/p/w300_and_h450_bestv2/3Gkb6jm6962ADUPaCBqzz9CTbn9.jpg"  // Twilight
    ];

    /* Update Movie Titles and Images, and store current rating */
    const [ ratings, setRatings ] = useState([]); // init submitted ratings array
    const [currentChoice, setCurrentChoice] = useState(null); // init each rating individually stored here

    // Add this missing state variable
    const [shouldFetchRecs, setShouldFetchRecs] = useState(false);

    // stores each rating they give as a single value
    const updateChoice = (currentRating) => {
        setCurrentChoice(updated => currentRating);
    }
    // adds currentChoice value to array of ratings
    const addRating = (newRating) => {
        setRatings(prev => [...prev, newRating]);
    }

    // boolean false if at the end of movie list
    const [reviewing, setReviewing] = useState(true);
    const [i, setI] = useState(0);  // init index for titles and images
    // increment i, update array, and turn boolean to false when at end of list
    function increment(e) {
        e.preventDefault()
        setI(i + 1);
        addRating(currentChoice);
        
        // Only when we reach the end of the list
        if (i >= titles.length - 1) {
            setReviewing(false);
            setShouldFetchRecs(true); // Trigger the API call only now
        }
        
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OMDB API implementation
    function ShowRecs({ shouldFetch }) {
        const [recommendations, setRecommendations] = useState([]);
        const [loading, setLoading] = useState(false);
        const [error, setError] = useState(null);
        const [hasFetched, setHasFetched] = useState(false);

        const getRecommendations = async () => {
            setLoading(true);
            setError(null);
            
            try {
                // Prepare user ratings data - exclude 1-star ratings
                const userRatings = ratings.map((rating, index) => ({
                    title: titles[index],
                    rating: rating
                })).filter(item => item.rating > 0 && item.rating !== 1);

                console.log('Sending ratings:', userRatings);

                const response = await fetch('http://localhost:5001/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        ratings: userRatings
                    })
                });

                if (!response.ok) {
                    throw new Error('Failed to get recommendations');
                }

                const data = await response.json();
                console.log('Received recommendations:', data.recommendations);
                setRecommendations(data.recommendations || []);
                setHasFetched(true);
            } catch (err) {
                setError(err.message);
                console.error('Error getting recommendations:', err);
            } finally {
                setLoading(false);
            }
        };

        // Only fetch when shouldFetch is true and we haven't fetched yet
        useEffect(() => {
            if (shouldFetch && !hasFetched) {
                getRecommendations();
            }
        }, [shouldFetch, hasFetched]);

        if (loading) {
            return (
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <h2>Getting your recommendations...</h2>
                    <p>Please wait while we analyze your ratings...</p>
                </div>
            );
        }

        if (error) {
            return (
                <div className="error-container">
                    <h2>Error getting recommendations</h2>
                    <p>{error}</p>
                    <button onClick={getRecommendations} className="retry-button">Try Again</button>
                </div>
            );
        }

        return (
            <div className="recommendations-container">
                <h2 className="recommendations-title">🎬 We recommend:</h2>
                {recommendations.length > 0 ? (
                    <div className="recommendations-list">
                        {recommendations.map((movie, index) => (
                            <div key={index} className="recommendation-item">
                                <span className="recommendation-number">{index + 1}.</span>
                                <span className="recommendation-title">{movie.title}</span>
                                <span className="recommendation-rating">⭐ {movie.predicted_rating.toFixed(1)}</span>
                                <span className="recommendation-genres">{movie.genres}</span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="no-recommendations">
                        <p>No recommendations available. Please rate some movies first!</p>
                    </div>
                )}
            </div>
        )
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////





    /*************************************************************************************************************************************************************************/
    
    return (
        <div className="survey">
            <div className={reviewing ? "main-active" : "main-inactive"}> 
                <div className="survey-header">
                    <h1 className="rate-title">Rate this movie</h1>
                    <div className="progress-bar">
                        <div 
                            className="progress-fill" 
                            style={{width: `${((i + 1) / titles.length) * 100}%`}}
                        ></div>
                    </div>
                    <p className="progress-text">{i + 1} of {titles.length}</p>
                </div>
                
                <div className="movie-display">
                    <div className="movie-poster-container">
                        <img id="img-display" src={images[i]} alt={titles[i]} />
                    </div>
                    <h2 id="title-display" className="movie-title-display">{titles[i]}</h2>
                </div>
                
                <div className="rating-section">
                    <StarRate onRate={updateChoice}/>
                    <button onClick={increment} className="submit-button">
                        {i >= titles.length - 1 ? 'Get Recommendations' : 'Next Movie'}
                    </button>
                </div>
            </div>

            <div className={reviewing ? "results-inactive" : "results-active"}> 
                <ShowRecs shouldFetch={shouldFetchRecs} /> 
            </div>

            <div className="ratings-sidebar">
                <h3>Your Ratings ({ratings.length})</h3>
                <div className="ratings-list">
                    {ratings.map((rating, idx) => (
                        <div key={idx} className="rating-item">
                            <span className="movie-name">{titles[idx]}</span>
                            <span className="rating-stars">
                                {'⭐'.repeat(rating)}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default Survey;
