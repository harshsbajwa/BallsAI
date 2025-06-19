import { PageHeader } from "@/components/page-header";
import { PlayerSearch } from "@/components/player-search";
import { TodaysGames } from "@/components/todays-games";
import { LeagueLeaders } from "@/components/league-leaders";
import { GamePredictor } from "@/components/game-predictor";
import { HeadToHeadAnalytics } from "@/components/head-to-head";

export default function Home() {
  return (
    <main>
      <div className="container mx-auto px-4 py-8">
        <PageHeader
          title="Welcome to BallsAI"
          description="Your ultimate hub for real-time NBA data, advanced analytics, and game predictions."
        />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
          <div className="lg:col-span-2 space-y-8">
            <TodaysGames />
            <LeagueLeaders />
          </div>
          <div className="space-y-8">
            <PlayerSearch />
            <GamePredictor />
            <HeadToHeadAnalytics /> {/* Add new component here */}
          </div>
        </div>
      </div>
    </main>
  );
}